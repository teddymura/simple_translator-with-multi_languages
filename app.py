from flask import Flask, request, jsonify, render_template
from gtts import gTTS
from googletrans import Translator
import whisper
import os
import uuid
import glob

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

translator = Translator()
model = whisper.load_model("base")  # CPU用モデル

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/api/translate_tts", methods=["POST"])
def translate_tts():
    audio = request.files.get("audio")
    target_lang = request.form.get("target_lang", "ja")  # ここで 'it' や 'vi' も対応

    if not audio:
        return jsonify({"error": "No audio file"}), 400

    # 一時ファイルとして保存
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(filepath)

    try:
        # Whisper で文字起こし
        result = model.transcribe(filepath)
        recognized_text = result["text"]

        # 翻訳
        translated = translator.translate(recognized_text, dest=target_lang)
        translated_text = translated.text

        # gTTS で音声生成
        tts_filename = f"{uuid.uuid4()}.mp3"
        tts_path = os.path.join("static", tts_filename)
        tts = gTTS(translated_text, lang=target_lang)
        tts.save(tts_path)

        # --- 古い mp3 を整理（最新5つだけ残す） ---
        mp3_files = sorted(glob.glob("static/*.mp3"), key=os.path.getmtime, reverse=True)
        for old_file in mp3_files[5:]:
            try:
                os.remove(old_file)
            except PermissionError:
                print("Warning: could not delete file, skipping:", old_file)

    finally:
        # 使い終わった音声ファイルは削除
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except PermissionError:
                print("Warning: could not delete file, skipping:", filepath)

    return jsonify({
        "recognized_text": recognized_text,
        "translated_text": translated_text,
        "audio_file": "/" + tts_path.replace("\\", "/")
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
