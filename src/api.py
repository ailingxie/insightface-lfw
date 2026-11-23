import base64
import tempfile
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from src.face_database import FaceDatabase
from src.face_utils import get_recognizer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 单请求上限 20MB

recognizer = get_recognizer()
db = FaceDatabase(recognizer=recognizer)


def decode_base64_image(b64: str):
    """把 base64（支持 data:image/...;base64, 前缀）解成 BGR 图像。"""
    if not b64:
        return None
    if "," in b64[:64] and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        data = base64.b64decode(b64)
    except Exception:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img


@app.get("/health")
def health():
    return jsonify({"status": "ok", "config": recognizer.summary(), "database": db.stats()})


@app.post("/recognize")
def recognize():
    """入参 {image: base64, topk?: int}，出参人脸列表 + Top-K 候选。"""
    data = request.get_json(silent=True) or {}
    img = decode_base64_image(data.get("image"))
    if img is None:
        return jsonify({"error": "No valid image provided"}), 400

    topk = int(data.get("topk", 1))
    results = []
    for face in recognizer.extract_features_from_image(img):
        name, score = db.recognize_face(face.embedding)
        results.append({
            "name": name,
            "confidence": round(float(score), 4),
            "bbox": face.bbox.astype(int).tolist(),
            "landmarks": face.kps.astype(float).tolist(),
            "det_score": round(float(face.det_score), 4),
            "topk": [{"name": n, "score": round(s, 4)} for n, s in db.search(face.embedding, topk=topk)],
        })

    return jsonify({"face_count": len(results), "results": results})


@app.post("/add_person")
def add_person():
    """入参 {name: str, images: [base64, ...]}，注册或更新一个身份。"""
    data = request.get_json(silent=True) or {}
    person_name = (data.get("name") or "").strip()
    images_base64 = data.get("images") or []
    if not person_name or not images_base64:
        return jsonify({"error": "Missing name or images"}), 400

    # 用系统临时目录，避免原来写死 /tmp 在 Windows 上直接失败
    paths = []
    tmp_dir = tempfile.mkdtemp(prefix="face_api_")
    for i, b64 in enumerate(images_base64):
        img = decode_base64_image(b64)
        if img is None:
            continue
        path = os.path.join(tmp_dir, f"{i}.jpg")
        cv2.imwrite(path, img)
        paths.append(path)

    ok = db.add_person(person_name, paths)
    if not ok:
        return jsonify({"success": False, "message": f"{person_name} 的图片中未检测到人脸"}), 422

    return jsonify({
        "success": True,
        "name": person_name,
        "samples": db.database[person_name]["samples"],
        "total_persons": len(db.database),
    })


@app.get("/list_persons")
def list_persons():
    persons = db.get_all_persons()
    return jsonify({"count": len(persons), "persons": persons})


if __name__ == "__main__":
    print(f"{recognizer.summary()} | {db.stats()}")
    app.run(host="0.0.0.0", port=5000, debug=False)
