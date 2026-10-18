import os
import json
from flask import Flask, request, jsonify, send_from_directory
from google import genai
from google.genai import types
from PIL import Image
import io

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/extract-words', methods=['POST'])
def extract_words():
    """
    接收前端上傳的多張單字表圖片，使用 Gemini 3.8 Flash 擷取所有單字，
    並自動用「台灣小學三年級」的英文能力造出挖空練習句 ({blank})。
    """
    api_key = request.form.get('api_key') or os.environ.get('GEMINI_API_KEY')
    model_name = request.form.get('model_name') or 'gemini-3.5-flash-lite'
    if not api_key:
        return jsonify({"error": "請提供 GEMINI_API_KEY"}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "未收到任何圖片檔案"}), 400

    client = genai.Client(api_key=api_key)

    contents = []
    prompt = """
你是一位專業的小學三年級英文老師。請仔細辨識上傳的所有圖片中的每一個英文單字、詞性(POS)與中文解釋。
並為每一個單字造一個「適合台灣小學三年級學生理解」的簡單、生活化英文例句。
例句中出現該單字的位置請一律用 {blank} 取代（保持單字原型，方便學生進行克漏字填空練習）。

請嚴格輸出 JSON Array 格式：
[
  {
    "word": "apple",
    "pos": "n.",
    "zh": "蘋果",
    "sentence": "I eat a sweet red {blank} every day."
  }
]
"""
    contents.append(prompt)

    for f in files:
        img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes))
        contents.append(img)

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    try:
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").removeprefix("json").strip()
        words_list = json.loads(raw_text)
        return jsonify({"words": words_list})
    except Exception as e:
        return jsonify({"error": f"解析 JSON 失敗: {str(e)}", "raw": response.text}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 互動式單字練習 App 已啟動：http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
