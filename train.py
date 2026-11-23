import os
import argparse
import glob
from config import Config
from src.face_database import FaceDatabase
from src.face_utils import get_recognizer

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def collect_images(person_dir):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(sorted(glob.glob(os.path.join(person_dir, ext))))
    return paths


def train_known_faces(append=False):
    """扫描 data/known_faces/<姓名>/ 下的照片，为每个身份生成一条平均特征入库。

    - 默认整库重建，但只在全部处理完才落盘（中途崩了不会毁掉旧库）
    - --append 只补录库里还没有的人
    """
    recognizer = get_recognizer()
    db = FaceDatabase(recognizer=recognizer, auto_save=False)
    if not append:
        db.database = {}       # 只在内存里清空，最后统一保存
        print("已清空内存中的旧库，开始全量重建")

    if not os.path.isdir(Config.KNOWN_FACES_DIR):
        print(f"目录不存在: {Config.KNOWN_FACES_DIR}")
        return

    person_dirs = [d for d in sorted(os.listdir(Config.KNOWN_FACES_DIR))
                   if os.path.isdir(os.path.join(Config.KNOWN_FACES_DIR, d))]

    ok, failed = 0, []
    for person_name in person_dirs:
        image_paths = collect_images(os.path.join(Config.KNOWN_FACES_DIR, person_name))
        if not image_paths:
            print(f"[跳过] {person_name}: 目录内没有图片")
            continue

        if db.add_person(person_name, image_paths, keep_existing=append):
            ok += 1
            info = db.database[person_name]
            print(f"[OK] {person_name}: 照片 {len(image_paths)} 张，有效样本 {info['samples']} 张")
        else:
            failed.append(person_name)
            print(f"[失败] {person_name}: 所有照片都没检测到人脸")

    db.save_database()
    print(f"\n注册完成：成功 {ok} 人，失败 {len(failed)} 人")
    if failed:
        print("失败名单:", ", ".join(failed))
    print(db.stats())
    print(f"库文件: {Config.FACE_DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建人脸特征库")
    parser.add_argument("--append", action="store_true", help="增量补录，不清空旧库")
    args = parser.parse_args()
    print(f"配置: {get_recognizer().summary()}")
    train_known_faces(append=args.append)
