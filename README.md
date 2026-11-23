# InsightFace + LFW 人脸识别系统

基于 InsightFace（buffalo_l 模型包）与 LFW 人脸数据集构建的**人脸检测 + 特征提取 + 1:N 身份识别**系统。支持命令行单图/批量识别、摄像头实时识别、Gradio 网页界面、Flask REST 接口四种使用方式。

---

## 1. 功能一览

| 功能 | 入口 | 说明 |
| --- | --- | --- |
| 构建人脸库 | `train.py` | 扫描 `data/known_faces/<姓名>/`，每人多张照片平均成一条特征入库 |
| 单图/批量识别 | `detect.py` | 支持单张图片或整个目录，输出标注图 + 相似度 + Top-K 候选 |
| 摄像头实时识别 | `video_recognition.py` | 逐帧检测并显示 FPS、已注册人数 |
| 网页界面 | `web_ui.py` | Gradio：在线识别、在线注册、人员表格、删除人员 |
| REST 接口 | `src/api.py` | Flask：`/recognize`、`/add_person`、`/list_persons`、`/health` |

---

## 2. 目录结构

```
insightface+lfw/
├── config.py                 # 全局配置：路径、模型、设备、阈值
├── train.py                  # 建库脚本（全量重建 / --append 增量）
├── detect.py                 # 单图 / 批量识别
├── video_recognition.py      # 摄像头实时识别
├── web_ui.py                 # Gradio 网页界面
├── requirements.txt
├── data/
│   ├── known_faces/<姓名>/   # 已知身份样本（LFW 裁剪后的照片）
│   ├── unknown_faces/        # 待识别照片
│   └── face_database.pkl     # 人脸特征库（身份 → 512 维平均特征）
├── models/
│   └── buffalo_l/            # 模型包：det_10g / w600k_r50 / 2d106det / 1k3d68 / genderage
├── outputs/                  # 识别结果图
└── src/
    ├── face_utils.py         # FaceAnalysis 封装：检测、特征提取、绘框、余弦相似度
    ├── face_database.py      # 人脸库：增删查、矩阵化 Top-K 检索、安全落盘
    └── api.py                # Flask 服务
```

---

## 3. 环境准备

本机已验证可用的解释器：`C:\Anaconda\envs\face_env\python.exe`（Python 3.8.20）。

```powershell
# 1) 安装依赖
C:\Anaconda\envs\face_env\python.exe -m pip install -r requirements.txt

# 2) 首次运行会自动下载 buffalo_l 模型包（约 275MB）到 models/buffalo_l/
#    也可以手动预置该目录，离线环境直接拷贝 models/ 即可
```

### 三个必须注意的坑（本机实测踩过）

1. **不要在 conda 环境里混用 MKL 版 numpy/scipy 和 PyPI 版**。
   conda 的 numpy/scipy 走 `Library\bin` 下的 Intel MKL + `libiomp5md.dll`，而 PyPI 的 scipy 自带 OpenBLAS 轮子，同一进程里两套 BLAS/OpenMP 会让 Python **直接崩溃**（错误码 `0xC06D007F`，表现为导入 `insightface` 时无 Traceback 闪退）。本项目依赖统一走 PyPI 轮子：

   ```powershell
   python -m pip install --force-reinstall --no-deps numpy==1.24.4 scipy==1.10.1
   ```

2. **GPU 不是装上显卡就能用**。`onnxruntime-gpu 1.16` 需要 CUDA 11.8 + cuDNN 8.x。若只有 `CUDA_PATH` 环境变量而缺 cuDNN，会报
   `CUDA_PATH is set but CUDA wasn't able to be loaded`。
   `src/face_utils.py` 已做了**自动回退**：GPU 初始化失败会打一行警告并改用 CPU，不会让程序中断。

3. **`ctx_id` 别写死**。原版代码写 `ctx_id=1`（第二块 GPU），单卡机器上直接建 session 失败。现在由 `Config.ctx_id()` 统一推导：有 CUDA 用 `GPU_ID`，否则 `-1`（CPU）。

4. **Windows 上 `cv2.imread` 读不了中文路径**。本项目路径含"学习"，`cv2.imread` 会**静默返回 None**，表现为"所有人脸都检测不到"却不报错。已统一改用 `src/face_utils.imread / imwrite`（`np.fromfile + cv2.imdecode`）。这是最难查的一类坑：不崩溃、不报错、只是结果全空。

5. **LFW 样本是 112×96 的小裁剪图，640 检测尺寸检不出脸**。实测同一张 `Susan_Collins_0001.jpg`：`det_size=640 → 0 张脸`，`det_size=320 → 1 张脸（0.794）`，`160 → 0.727`。原因是这类"脸占满画面、没有背景上下文"的图在大尺寸下反而过不了检测器。所以 `extract_features_from_image` 改成**自适应尺寸**：先按 `DET_SIZE` 试，检不到再按 `DET_SIZE_FALLBACK = [(320,320),(160,160)]` 逐级重试。

---

## 4. 快速开始

```powershell
cd C:\学习\PycharmProjects\insightface+lfw

# 构建人脸库（全量重建，建议每次新增人物后跑一次）
python train.py

# 只补录库里还没有的人
python train.py --append

# 识别单张图片（弹窗显示 + 保存到 outputs/）
python detect.py --image data\unknown_faces\xxx.jpg

# 不弹窗（服务器/远程环境）、打印 Top-3 候选、自定义阈值
python detect.py --image data\unknown_faces\xxx.jpg --no-show --topk 3 --threshold 0.5

# 批量识别整个目录
python detect.py --image data\unknown_faces

# 摄像头实时识别（每 3 帧推理一次）
python video_recognition.py --camera 0 --interval 3

# 网页界面 → http://127.0.0.1:7860
python web_ui.py

# REST 服务 → http://127.0.0.1:5000
python -m src.api
```

### 接口示例

```powershell
# 健康检查
curl http://127.0.0.1:5000/health

# 识别（image 为 base64，可带 data:image/jpeg;base64, 前缀）
curl -X POST http://127.0.0.1:5000/recognize -H "Content-Type: application/json" `
     -d '{\"image\":\"<base64>\",\"topk\":3}'
```

返回：

```json
{
  "face_count": 1,
  "results": [{
    "name": "Obama",
    "confidence": 0.87,
    "bbox": [866, 192, 1330, 867],
    "landmarks": [[...], ...],
    "det_score": 0.91,
    "topk": [{"name": "Obama", "score": 0.87}, {"name": "xxx", "score": 0.21}]
  }]
}
```

---

## 5. 核心原理

整条链路分四步，全部在 `src/face_utils.py` + `src/face_database.py` 里：

1. **人脸检测**：`det_10g.onnx`（RetinaFace/SCRFD 结构），输出人脸框 + 5 个关键点 + 检测置信度。`DET_THRESH` 控制误检，`DET_SIZE` 影响小脸召回（本项目 640×640）。
2. **对齐与归一化**：用 5 点关键点做相似变换，把人脸摆正、裁剪缩放到 112×112，消除姿态和尺度差异。
3. **特征提取**：`w600k_r50.onnx`（ArcFace，ResNet50 + WebFace600K 训练），输出 **512 维**特征向量，同一身份向量夹角小、不同身份夹角大。
4. **1:N 检索**：
   - 建库时同一人多张照片的特征取**均值再 L2 归一化**，降低单张样本噪声（`face_database.add_person`）；
   - 识别时把库内所有向量堆成 `(N, 512)` 矩阵，一次矩阵乘法得全部余弦相似度（`FaceDatabase.search`），而不是逐条 `np.dot` 循环；
   - 取最高分，低于 `REC_THRESH`（默认 0.4）判为 `Unknown`，否则输出身份 + 相似度。

```
输入图 → 检测(SCRFD) → 5点对齐 → 112×112 → ArcFace → 512维 → 余弦相似度 → 阈值判定 → 身份/Unknown
```

---

## 6. 参数说明（`config.py`）

| 参数 | 默认 | 作用 | 调参建议 |
| --- | --- | --- | --- |
| `MODEL_NAME` | `buffalo_l` | 模型包，越大越准越慢 | 演示用 `buffalo_s` 更快，精度略降 |
| `DET_THRESH` | 0.5 | 检测置信度阈值 | 漏检多就降到 0.3，误检多就升到 0.6 |
| `DET_SIZE` | (640, 640) | 检测输入尺寸 | 小脸/多人场景保持 640；追求速度用 320 |
| `DET_SIZE_FALLBACK` | [(320,320),(160,160)] | 主尺寸检不到脸时的降级重试 | 样本是紧贴人脸的裁剪图时靠它兜底 |
| `REC_THRESH` | 0.4 | 余弦相似度阈值 | 认错人（误接受）就升到 0.5；总报 Unknown 就降到 0.35 |
| `USE_GPU` | `auto` | `auto/on/off` | 无可用 CUDA 时保持 auto，代码会自动回退 CPU |
| `FACE_DB_PATH` | `data/face_database.pkl` | 特征库位置 | 换库/多库测试时改这里 |

环境变量同样可覆盖：`FACE_GPU=off`、`FACE_GPU_ID=0`。

---

## 7. 实测结果

| 项目 | 结果 |
| --- | --- |
| 已注册身份 | **262 人**（LFW 子集，来自 `data/known_faces/`，无脏数据残留） |
| 有效样本 | **648 张**，每人平均约 2.5 张 |
| 同人识别 | Obama 两张不同照片分别 0.870 / 0.867，均判对 |
| 区分度 | Obama 最相似的干扰项仅 0.166 / 0.147，首位与次位差距 >0.6，几乎没有混淆空间 |
| 服务入口 | Flask `/health`、`/recognize` 实测 200；Gradio 页面实测 HTTP 200 |
| 推理设备 | CPU（onnxruntime-gpu 已装，但 `onnxruntime_providers_cuda.dll` 加载失败 → 自动回退） |
| 单图耗时 | CPU 下约 0.2–0.4s（640 检测尺寸；小图降级重试会略慢） |

> **注意**：库里只有一张样本的身份，自测相似度必然是 1.000（查询向量和库向量就是同一张图算出来的），这不代表真实精度。要报准确率，必须用**未入库的照片**或 LFW 官方 6000 对验证集跑 ROC/TAR@FAR，别把 1.000 当成绩。

---

## 8. 常见问题（FAQ）

**Q：`import insightface` 直接闪退，没有任何报错？**
A：说明是原生层崩溃（DLL 冲突），不是 Python 异常。先按第 3 节第 1 条统一 numpy/scipy 到 PyPI 轮子；排查用 `python -X faulthandler -c "import insightface"` 看崩在哪个模块。

**Q：`CUDA_PATH is set but CUDA wasn't able to be loaded`？**
A：装 CUDA 11.8 + cuDNN 8.x 并把 `bin` 加入 PATH；或者什么都不做，代码会自动用 CPU。

**Q：注册总失败 / 库里样本数 0？**
A：照片里人脸太小、太糊、侧脸过度。换清晰正脸照，或把 `DET_SIZE` 提到 640、`DET_THRESH` 降到 0.3。

**Q：多人合照识别时认错人？**
A：建库已改为取"面积最大的人脸"（`largest_face_embedding`），避免取到路人；合照识别时提高 `REC_THRESH`。

**Q：`pip` 装的包版本乱？**
A：本机 `pip` 指向华为云镜像，装不动可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

---

## 9. 可继续扩展的方向

- **活体检测**：加眨眼/张嘴动作或静默活体模型，防止照片翻拍攻击。
- **精度评估**：接 LFW 6000 对验证集，画 ROC、算 TAR@FAR，把"准确率"变成可复现指标。
- **底库规模化**：向量数量上到万级后换 FAISS（IndexFlatIP）/ HNSW，替代当前的全量矩阵乘法。
- **工程化**：pkl 换 SQLite/Postgres、多进程 worker、接口加鉴权与限流、模型量化（INT8）提速。
