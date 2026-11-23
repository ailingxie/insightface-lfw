import os
import argparse
import cv2
from config import Config
from src.face_database import FaceDatabase
from src.face_utils import get_recognizer, imread, imwrite

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def recognize_image(image_path, db, recognizer, show=True, topk=1, save=True):
    """识别单张图片，返回 [(name, score), ...] 与结果图。"""
    img = imread(image_path)
    if img is None:
        print(f"[跳过] 无法读取图片: {image_path}")
        return [], None

    faces = recognizer.extract_features_from_image(img)
    if not faces:
        print(f"[无人脸] {os.path.basename(image_path)}")
        return [], None

    names = []
    for face in faces:
        name, score = db.recognize_face(face.embedding)
        names.append(f"{name} ({score:.3f})" if name != "Unknown" else "Unknown")

    result_img = recognizer.draw_faces(img, faces, names)

    if save:
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(Config.OUTPUT_DIR, f"{stem}_result.jpg")
        imwrite(out_path, result_img)
        print(f"[保存] {out_path}")

    for i, (face, label) in enumerate(zip(faces, names)):
        line = f"  人脸{i + 1}: {label}  bbox={face.bbox.astype(int).tolist()}  det={face.det_score:.2f}"
        if topk > 1:
            cand = " | ".join(f"{n}:{s:.3f}" for n, s in db.search(face.embedding, topk=topk))
            line += f"  Top{topk}={cand}"
        print(line)

    if show:
        cv2.imshow("Face Recognition", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return names, result_img


def main():
    parser = argparse.ArgumentParser(description="InsightFace 人脸识别 - 单图/批量")
    parser.add_argument("--image", required=True, help="图片路径，或一个图片目录")
    parser.add_argument("--no-show", action="store_true", help="不弹窗显示（服务器/远程环境用）")
    parser.add_argument("--no-save", action="store_true", help="不保存结果图")
    parser.add_argument("--threshold", type=float, default=None, help="相似度阈值，默认取 Config.REC_THRESH")
    parser.add_argument("--topk", type=int, default=1, help="同时打印 Top-K 候选")
    args = parser.parse_args()

    recognizer = get_recognizer()
    db = FaceDatabase(recognizer=recognizer)
    if args.threshold is not None:
        Config.REC_THRESH = args.threshold

    print(f"配置: {recognizer.summary()}")
    print(f"库: {db.stats()}")

    if os.path.isdir(args.image):
        files = [f for f in sorted(os.listdir(args.image)) if f.lower().endswith(IMG_EXTS)]
        print(f"批量识别：{len(files)} 张")
        for f in files:
            print(f"\n--- {f} ---")
            recognize_image(os.path.join(args.image, f), db, recognizer,
                            show=False, topk=args.topk, save=not args.no_save)
    else:
        recognize_image(args.image, db, recognizer,
                        show=not args.no_show, topk=args.topk, save=not args.no_save)


if __name__ == "__main__":
    main()
