import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config import Config

_recognizer = None


def imread(path):
    """读图片。

    cv2.imread 在 Windows 上无法处理含中文的路径（本项目路径里有"学习"），
    会静默返回 None，导致所有人脸都检测不到。这里改用 np.fromfile + imdecode。
    """
    if not path or not os.path.exists(path):
        return None
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def imwrite(path, image):
    """写图片，同样绕开中文路径问题。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buf.tofile(path)
    return True


def get_recognizer() -> "FaceRecognizer":
    """进程内单例：模型只加载一次，web_ui / api / detect 共用同一份。"""
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer()
    return _recognizer


class FaceRecognizer:
    """封装 insightface 的 FaceAnalysis，对外提供检测 + 特征提取 + 绘框。"""

    def __init__(self, det_thresh=None, det_size=None):
        self.det_thresh = Config.DET_THRESH if det_thresh is None else det_thresh
        self.primary_size = tuple(Config.DET_SIZE if det_size is None else det_size)
        self.device = "CPU"
        self._apps = {}          # det_size -> FaceAnalysis，按需构建并缓存
        self.app = self._app_for(self.primary_size)

    # ---------- 内部：按检测尺寸构建（失败自动回退 CPU） ----------
    def _app_for(self, det_size):
        det_size = tuple(det_size)
        if det_size in self._apps:
            return self._apps[det_size]

        try:
            app = self._build(det_size, Config.providers(), Config.ctx_id())
            if Config.use_gpu():
                self.device = f"GPU:{Config.GPU_ID}"
        except Exception as exc:
            if not Config.use_gpu():
                raise
            # 有 CUDA 环境变量但缺 cuDNN 等运行时，onnxruntime 会抛异常，这里回退 CPU
            print(f"[警告] GPU 初始化失败，已回退 CPU：{str(exc).splitlines()[0][:160]}")
            app = self._build(det_size, ["CPUExecutionProvider"], -1)

        self._apps[det_size] = app
        return app

    def _build(self, det_size, providers, ctx_id):
        app = FaceAnalysis(name=Config.MODEL_NAME, root=Config.MODEL_ROOT, providers=providers)
        app.prepare(ctx_id=ctx_id, det_thresh=self.det_thresh, det_size=tuple(det_size))
        return app

    def summary(self) -> str:
        """实际生效的配置（GPU 失败回退后这里显示 CPU，不会说谎）。"""
        return (f"model={Config.MODEL_NAME} device={self.device} "
                f"det={self.primary_size} rec_thresh={Config.REC_THRESH}")

    # ---------- 特征提取 ----------
    def extract_features_from_image(self, img_array):
        """检测 + 对齐 + 特征提取，返回 insightface 的 Face 对象列表。

        LFW 里很多样本是 112×96 的小裁剪图，在 640 检测尺寸下一个脸都检不出来，
        所以按 Config.DET_SIZE_FALLBACK 逐级降尺寸重试，拿到结果就返回。
        """
        if img_array is None:
            return []
        if img_array.ndim == 2:          # 灰度图转三通道
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        img_array = np.ascontiguousarray(img_array)

        sizes = [self.primary_size]
        for s in Config.DET_SIZE_FALLBACK:
            if tuple(s) not in sizes:
                sizes.append(tuple(s))

        for size in sizes:
            faces = self._app_for(size).get(img_array)
            if faces:
                return faces
        return []

    def extract_face_features(self, image_path):
        """从图片路径提取特征，返回 dict 列表（bbox / kps / det_score / embedding）。"""
        img = imread(image_path)
        if img is None:
            return []
        return [{
            "embedding": face.embedding,
            "bbox": face.bbox.astype(int),
            "kps": face.kps,
            "det_score": float(face.det_score),
        } for face in self.extract_features_from_image(img)]

    def largest_face_embedding(self, image_path):
        """取画面中面积最大的人脸特征。

        原来直接取 faces[0]，多人合照时可能取到路人，导致平均特征被污染。
        """
        img = imread(image_path)
        if img is None:
            return None
        faces = self.extract_features_from_image(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.embedding

    # ---------- 绘制 ----------
    def draw_faces(self, image, faces, names=None):
        """在图像上画人脸框、关键点和名字，返回新图像（不改原图）。"""
        img_draw = image.copy()
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            known = names and i < len(names) and names[i] != "Unknown"
            color = (0, 200, 0) if known else (0, 165, 255)
            cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            for kp in face.kps.astype(int):
                cv2.circle(img_draw, tuple(kp), 2, (0, 0, 255), -1)
            if names and i < len(names):
                y = max(bbox[1] - 10, 18)
                cv2.putText(img_draw, names[i], (bbox[0], y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return img_draw


def cosine_similarity(a, b) -> float:
    """归一化后的余弦相似度。"""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a / na, b / nb))
