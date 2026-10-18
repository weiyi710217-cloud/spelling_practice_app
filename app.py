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
    api_key = request.form.get('api_key') or os.environ.get('GEMINI_API_KEY')
    model_name = request.form.get('model_name') or 'gemini-3.5-flash-lite'
    
    if not api_key:
        return jsonify({"error": "請提供 GEMINI_API_KEY"}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "未收到任何圖片檔案"}), 400

    try:
        client = genai.Client(api_key=api_key)
        prompt = """
你是一位專業的小學三年級英文老師。請仔細辨識這張圖片中的每一個英文單字、詞性(POS)與中文解釋。
並為每一個單字造一個「適合台灣小學三年級學生理解」的簡單、生活化英文例句。
例句中出現該單字的位置請一律用 {blank} 取代（保持原型，方便學生作答）。

【重要格式要求】
請每一行輸出一個單字，欄位之間用三個直線「 ||| 」分隔，格式嚴格如下：
單字 ||| 詞性 ||| 中文解釋 ||| 包含 {blank} 的英文例句

範例：
persistent ||| adj. ||| 堅持不懈的/持續的 ||| The little boy is very {blank} and tries again and again.

請直接輸出單字資料行，不要加上任何說明文字：
"""

        all_text_results = []
        for f in files:
            img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))

            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192
                ),
            )
            if response.text:
                all_text_results.append(response.text)

        return jsonify({"raw_text": "\n".join(all_text_results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 App 已啟動：http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
