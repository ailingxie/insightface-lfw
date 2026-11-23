import os
import pickle
import numpy as np
from tqdm import tqdm
from src.face_utils import get_recognizer
from config import Config


class FaceDatabase:
    """人脸库：每个身份存一条 512 维归一化平均特征，识别用余弦相似度 + Top-K。"""

    def __init__(self, recognizer=None, db_path=None, auto_load=True, auto_save=True):
        # 默认复用全局单例，避免每个入口脚本各加载一份模型
        self.recognizer = recognizer or get_recognizer()
        self.db_path = db_path or Config.FACE_DB_PATH
        self.database = {}
        # 批量建库时先关掉逐条保存，全部处理完再落盘，避免中途失败写坏库
        self.auto_save = auto_save
        if auto_load:
            self.load_database()

    # ---------- 持久化 ----------
    def load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.database = pickle.load(f)
        return self.database

    def save_database(self):
        """先写临时文件再替换，避免进程被中断时把 pkl 写坏。"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        tmp = self.db_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.database, f)
        os.replace(tmp, self.db_path)

    # ---------- 写 ----------
    def add_person(self, person_name, image_paths, keep_existing=False):
        """把同一个人的多张照片平均成一条特征。

        keep_existing=True 时，已注册的人直接跳过（增量建库用）。
        """
        if person_name in self.database and keep_existing:
            return True

        embeddings = []
        for img_path in tqdm(image_paths, desc=f"提取特征 {person_name}", leave=False):
            emb = self.recognizer.largest_face_embedding(img_path)
            if emb is not None:
                embeddings.append(np.asarray(emb, dtype=np.float32))

        if not embeddings:
            return False

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm == 0:
            return False

        self.database[person_name] = {
            "embedding": (avg / norm).astype(np.float32),
            "samples": len(embeddings),
            "failed": len(image_paths) - len(embeddings),
            "image_paths": list(image_paths),
        }
        if self.auto_save:
            self.save_database()
        return True

    def remove_person(self, person_name):
        if person_name in self.database:
            del self.database[person_name]
            self.save_database()
            return True
        return False

    def reset(self):
        self.database = {}
        self.save_database()

    # ---------- 读 ----------
    def _matrix(self):
        """把 dict 展开成 (N, 512) 矩阵 + 名字列表，便于一次性矩阵乘法，比循环 dot 快。"""
        names, vecs = [], []
        for name, data in self.database.items():
            emb = np.asarray(data.get("embedding"), dtype=np.float32).ravel()
            if emb.size == 0:
                continue
            names.append(name)
            vecs.append(emb)
        if not names:
            return [], np.zeros((0, 0), dtype=np.float32)
        return names, np.vstack(vecs)

    def search(self, embedding, topk=1):
        """返回 [(name, similarity), ...]，按相似度降序。"""
        names, mat = self._matrix()
        if not names:
            return []
        query = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        scores = mat @ (query / norm)          # 库内向量已归一化，点积即余弦相似度
        order = np.argsort(-scores)[:max(1, topk)]
        return [(names[i], float(scores[i])) for i in order]

    def recognize_face(self, embedding, threshold=None):
        """识别单张人脸：高于阈值返回身份，否则返回 Unknown。"""
        threshold = Config.REC_THRESH if threshold is None else threshold
        hits = self.search(embedding, topk=1)
        if not hits:
            return "Unknown", 0.0
        name, score = hits[0]
        if score < threshold:
            return "Unknown", score
        return name, score

    def get_all_persons(self):
        return list(self.database.keys())

    def stats(self):
        samples = sum(int(d.get("samples", 0)) for d in self.database.values())
        return f"已注册 {len(self.database)} 人，共 {samples} 张有效样本"
