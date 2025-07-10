from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import base64

from utils.extractor import extract_text
from deepseek_client import DeepSeekClient
from whisper_utils.whisper_transcriber import WhisperTranscriber
from utils.excel_reader import ExcelReader
from rag_retriever import retrieve_top_k

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = DeepSeekClient()
transcriber = WhisperTranscriber()

# 工具函数：获取所有文件夹和对应文件
def get_folder_files():
    folder_files = {}
    for folder in os.listdir(UPLOAD_FOLDER):
        folder_path = os.path.join(UPLOAD_FOLDER, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            folder_files[folder] = files
    return folder_files

@app.route('/')
def index():
    folder_files = get_folder_files()
    folders = list(folder_files.keys())
    return render_template('index.html', folder_files=folder_files, folders=folders)

# ✅ 上传文件
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    folder = request.form['folder']
    target_folder = os.path.join(UPLOAD_FOLDER, folder)
    os.makedirs(target_folder, exist_ok=True)
    filepath = os.path.join(target_folder, file.filename)
    file.save(filepath)
    return redirect(url_for('index'))

# ✅ 创建文件夹
@app.route('/create_folder', methods=['POST'])
def create_folder():
    folder_name = request.form['folder_name']
    path = os.path.join(UPLOAD_FOLDER, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return redirect(url_for('index'))

# ✅ 删除文件
@app.route('/delete/<folder>/<filename>')
def delete_file(folder, filename):
    file_path = os.path.join(UPLOAD_FOLDER, folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('index'))

# ✅ 查看文件
@app.route('/view/<folder>/<filename>')
def view_file(folder, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, folder), filename)

# ✅ 下载文件
@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, folder), filename, as_attachment=True)

# ✅ 文本输入问答（RAG）
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    top_chunks = retrieve_top_k(question, k=5)
    context = "\n".join(top_chunks)
    prompt = f"根据以下资料回答问题：\n\n{context}\n\n问题：{question}"
    answer = model.ask(prompt)
    folder_files = get_folder_files()
    folders = list(folder_files.keys())
    return render_template("index.html",
                           answer=answer,
                           source_chunks=top_chunks,  # ✅ 显示引用片段
                           folder_files=folder_files,
                           folders=folders)

# ✅ 浏览器录音上传提问（RAG）
@app.route('/ask_from_voice', methods=['POST'])
def ask_from_voice():
    base64_audio = request.form['audio_blob']
    audio_data = base64.b64decode(base64_audio)
    audio_path = os.path.join(UPLOAD_FOLDER, 'audio', 'recorded.wav')
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    with open(audio_path, 'wb') as f:
        f.write(audio_data)

    question = transcriber.transcribe_audio(audio_path)
    top_chunks = retrieve_top_k(question, k=5)
    context = "\n".join(top_chunks)
    prompt = f"根据以下资料回答问题：\n\n{context}\n\n用户语音提问是：{question}"
    answer = model.ask(prompt)

    folder_files = get_folder_files()
    folders = list(folder_files.keys())
    return render_template("index.html",
                           answer=answer,
                           voice_text=question,
                           source_chunks=top_chunks,  # ✅ 显示引用片段
                           folder_files=folder_files,
                           folders=folders)

# ✅ 上传音频文件识别提问（RAG）
@app.route('/ask_from_audio', methods=['POST'])
def ask_from_audio():
    audio_file = request.files['audio_file']
    path = os.path.join(UPLOAD_FOLDER, 'audio', audio_file.filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio_file.save(path)

    question = transcriber.transcribe_audio(path)
    top_chunks = retrieve_top_k(question, k=5)
    context = "\n".join(top_chunks)
    prompt = f"根据以下资料回答问题：\n\n{context}\n\n用户上传的音频内容是：{question}"
    answer = model.ask(prompt)

    folder_files = get_folder_files()
    folders = list(folder_files.keys())
    return render_template("index.html",
                           answer=answer,
                           voice_text=question,
                           source_chunks=top_chunks,  # ✅ 显示引用片段
                           folder_files=folder_files,
                           folders=folders)

# ✅ 上传 Excel 文件识别提问（RAG）
@app.route('/ask_from_excel', methods=['POST'])
def ask_from_excel():
    excel_file = request.files['excel_file']
    excel_path = os.path.join(UPLOAD_FOLDER, 'excel', excel_file.filename)
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    excel_file.save(excel_path)

    user_question = request.form['excel_question']
    reader = ExcelReader(excel_path)
    headers, rows = reader.extract_data()

    if not headers:
        return "❌ 无法读取 Excel，请检查文件格式"

    structured_info = f"表头字段：{headers}\n前几行数据：{rows}"
    prompt_question = f"以下是用户上传的 Excel 表格信息：\n{structured_info}\n\n问题：{user_question}"

    top_chunks = retrieve_top_k(user_question, k=5)
    context = "\n".join(top_chunks)
    final_prompt = f"{context}\n\n{prompt_question}"
    answer = model.ask(final_prompt)

    folder_files = get_folder_files()
    folders = list(folder_files.keys())
    return render_template("index.html",
                           answer=answer,
                           source_chunks=top_chunks,  # ✅ 显示引用片段
                           folder_files=folder_files,
                           folders=folders)

if __name__ == '__main__':
    app.run(debug=True)
