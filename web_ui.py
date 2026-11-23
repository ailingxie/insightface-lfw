import os
import cv2
import numpy as np
import gradio as gr
from config import Config
from src.face_database import FaceDatabase
from src.face_utils import get_recognizer

recognizer = get_recognizer()
db = FaceDatabase(recognizer=recognizer)


def recognize_interface(image, threshold):
    """识别上传的图片，返回标注图和文字结果。"""
    if image is None:
        return None, "请先上传图片"

    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Gradio 传进来是 RGB

    faces = recognizer.extract_features_from_image(img)
    if not faces:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "未检测到人脸"

    names, lines = [], [f"检测到 {len(faces)} 张人脸（阈值 {threshold:.2f}）"]
    for i, face in enumerate(faces):
        name, score = db.recognize_face(face.embedding, threshold=threshold)
        top = db.search(face.embedding, topk=3)
        cand = " | ".join(f"{n}:{s:.3f}" for n, s in top)
        names.append(f"{name} ({score:.3f})" if name != "Unknown" else "Unknown")
        lines.append(f"人脸{i + 1}: {name}  相似度={score:.3f}  Top3={cand}")

    result = recognizer.draw_faces(img, faces, names)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), "\n".join(lines)


def add_person_interface(name, files):
    """把上传的多张照片注册成一个身份。files 是 Gradio 落盘的临时文件路径列表。"""
    name = (name or "").strip()
    if not name:
        return "请输入姓名"
    if not files:
        return "请选择至少一张图片"

    paths = [f if isinstance(f, str) else f.name for f in files]
    if not db.add_person(name, paths):
        return f"添加失败：{name} 的照片里都没检测到清晰人脸"

    info = db.database[name]
    return f"已添加 {name}：有效样本 {info['samples']} 张，失败 {info['failed']} 张"


def person_table():
    rows = [[n, d.get("samples", 0), d.get("failed", 0)] for n, d in db.database.items()]
    return sorted(rows, key=lambda r: r[0])


def remove_person_interface(name):
    if not name:
        return "请输入要删除的姓名"
    return f"已删除 {name}" if db.remove_person(name.strip()) else f"库里没有 {name}"


with gr.Blocks(title="人脸识别系统") as demo:
    gr.Markdown(f"# 🎭 人脸识别系统\n配置：`{recognizer.summary()}` ｜ 库：{db.stats()}")

    with gr.Tab("人脸识别"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="上传图片", type="numpy")
                threshold_slider = gr.Slider(0.1, 0.9, value=Config.REC_THRESH, step=0.01,
                                             label="识别阈值（越高越严格）")
                recognize_btn = gr.Button("开始识别", variant="primary")
            with gr.Column():
                image_output = gr.Image(label="识别结果")
                text_output = gr.Textbox(label="识别明细", lines=6)

        recognize_btn.click(fn=recognize_interface, inputs=[image_input, threshold_slider],
                            outputs=[image_output, text_output])

    with gr.Tab("添加人员"):
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(label="姓名")
                file_inputs = gr.File(label="选择多张图片", file_count="multiple", file_types=["image"])
                add_btn = gr.Button("注册", variant="primary")
            with gr.Column():
                add_result = gr.Textbox(label="注册结果", lines=3)

        add_btn.click(fn=add_person_interface, inputs=[name_input, file_inputs], outputs=add_result)

    with gr.Tab("已注册人员"):
        table = gr.Dataframe(headers=["姓名", "有效样本", "失败样本"], value=person_table())
        with gr.Row():
            refresh_btn = gr.Button("刷新列表")
            del_name = gr.Textbox(label="要删除的姓名", scale=2)
            del_btn = gr.Button("删除")

        refresh_btn.click(fn=person_table, inputs=[], outputs=table)
        del_btn.click(fn=remove_person_interface, inputs=del_name, outputs=add_result)
        del_btn.click(fn=person_table, inputs=[], outputs=table)

if __name__ == "__main__":
    # inbrowser=True：启动后自动拉起默认浏览器；show_error=True 让异常显示在页面上而不是只有控制台
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
