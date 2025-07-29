import os, uuid, json
from flask import Flask, request, jsonify

app = Flask(__name__)

SAVE_DIR = "/app/models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Text → Image
@app.route('/text2img', methods=['POST'])
def text2img():
    data = request.get_json(force=True)
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    # твоя логіка генерувати image = pipe(prompt).images[0]
    image = pipe(prompt).images[0]  # припускаємо pipe вже ініціалізовано вище
    fname = f"{uuid.uuid4().hex}.png"
    path = os.path.join(SAVE_DIR, fname)
    image.save(path)

    return jsonify({"output": fname})

# Статика моделей
@app.route('/models/<path:filename>')
def serve_models(filename):
    return app.send_from_directory(SAVE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)