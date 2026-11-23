import os
import onnxruntime as ort


class Config:
    """全局配置：路径、模型、设备、阈值集中在这里，其他脚本只读不改。"""

    # ---------- 路径 ----------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
    UNKNOWN_FACES_DIR = os.path.join(DATA_DIR, "unknown_faces")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    # FaceAnalysis 的 root 指向 BASE_DIR 时，模型会落在 models/<MODEL_NAME>
    MODEL_ROOT = BASE_DIR
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # ---------- 模型 ----------
    MODEL_NAME = "buffalo_l"      # 可选: buffalo_l / buffalo_m / buffalo_s
    DET_THRESH = 0.5              # 检测置信度阈值，低于该值的人脸框丢弃
    DET_SIZE = (640, 640)         # 检测输入尺寸，越大越小脸越稳，但更慢
    # 主尺寸检测不到脸时按顺序降尺寸重试（LFW 的 112×96 小裁剪图在 640 下检不出）
    DET_SIZE_FALLBACK = [(320, 320), (160, 160)]
    REC_THRESH = 0.4              # 余弦相似度阈值，低于该值判为 Unknown

    # ---------- 设备 ----------
    # FACE_GPU=auto/on/off 环境变量可覆盖：auto 有 CUDA 就用，没有自动退回 CPU
    USE_GPU = os.environ.get("FACE_GPU", "auto").lower()
    GPU_ID = int(os.environ.get("FACE_GPU_ID", "0"))

    # ---------- 数据库 ----------
    FACE_DB_PATH = os.path.join(DATA_DIR, "face_database.pkl")

    # ---------- 图像 ----------
    FACE_SIZE = (112, 112)        # 人脸对齐后尺寸（ArcFace 标准输入）

    # ---------- 派生配置 ----------
    @classmethod
    def use_gpu(cls) -> bool:
        """是否启用 GPU 推理。auto 模式下探测 onnxruntime 是否有 CUDA provider。"""
        if cls.USE_GPU == "off":
            return False
        if cls.USE_GPU == "on":
            return True
        return "CUDAExecutionProvider" in ort.get_available_providers()

    @classmethod
    def providers(cls) -> list:
        """返回 onnxruntime 的执行器列表。"""
        if cls.use_gpu():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @classmethod
    def ctx_id(cls) -> int:
        """insightface 的设备号：>=0 表示用第 n 块 GPU，-1 表示 CPU。

        原代码写死 ctx_id=1，在没有双卡/没有 CUDA 的机器上会直接建 session 失败。
        """
        return cls.GPU_ID if cls.use_gpu() else -1

    @classmethod
    def describe(cls) -> str:
        device = f"GPU:{cls.GPU_ID}" if cls.use_gpu() else "CPU"
        return f"model={cls.MODEL_NAME} device={device} rec_thresh={cls.REC_THRESH}"
