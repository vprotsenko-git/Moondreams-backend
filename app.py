import os, uuid, json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionPipeline

app = Flask(__name__)
CORS(app)

# Де зберігати результати
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Ініціалізація моделі (одноразово)
device = "cuda" if os.getenv("USE_CUDA") == "1" else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype="auto"
).to(device)

# Генерація картинки
@app.route('/text2img', methods=['POST'])
def text2img():
    data = request.get_json(force=True)
    prompt = data.get('prompt','').strip()
    if not prompt:
        return jsonify(error="Missing prompt"), 400

    image = pipe(prompt).images[0]
    fname = f"{uuid.uuid4().hex}.png"
    path = os.path.join(SAVE_DIR, fname)
    image.save(path)
    return jsonify(output=fname)

# Статика з models/
@app.route('/models/<path:fn>')
def static_model(fn):
    return send_from_directory(SAVE_DIR, fn)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)