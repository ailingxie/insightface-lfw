import time
import argparse
import cv2
from config import Config
from src.face_database import FaceDatabase
from src.face_utils import get_recognizer


def realtime_recognition(camera_id=0, interval=3, mirror=True):
    """摄像头实时识别。

    interval: 每 N 帧做一次识别（推理较慢时保证画面流畅，识别结果做缓存复用）
    原代码用 `if frame_count % 2 == 0: continue`，会直接丢掉一半帧不显示，看起来像掉帧卡顿。
    """
    recognizer = get_recognizer()
    db = FaceDatabase(recognizer=recognizer)

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}，检查是否被其他程序占用")
        return

    print(f"{recognizer.summary()} | {db.stats()} | 按 q 退出")

    last_faces, last_names, last_fps = [], [], 0.0
    frame_count = 0
    t_last = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取视频帧失败")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        # 每 interval 帧刷新一次识别结果，其余帧复用上次结果
        if frame_count % interval == 0:
            t0 = time.time()
            faces = recognizer.extract_features_from_image(frame)
            names = []
            for face in faces:
                name, score = db.recognize_face(face.embedding)
                names.append(f"{name} ({score:.3f})" if name != "Unknown" else "Unknown")
            last_faces, last_names = faces, names
            last_fps = 1.0 / max(time.time() - t0, 1e-6)
        frame_count += 1

        result = recognizer.draw_faces(frame, last_faces, last_names)

        now = time.time()
        loop_fps = 1.0 / max(now - t_last, 1e-6)
        t_last = now
        cv2.putText(result, f"Loop FPS: {loop_fps:.1f} | Infer: {last_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result, f"Registered: {len(db.database)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Real Time", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实时摄像头人脸识别")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--interval", type=int, default=3, help="每 N 帧识别一次")
    parser.add_argument("--no-mirror", action="store_true", help="不做镜像翻转")
    args = parser.parse_args()
    realtime_recognition(args.camera, args.interval, mirror=not args.no_mirror)
