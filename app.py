import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)
CORS(app)

SAVE_DIR = "/app/models"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

@app.route('/api/text2img', methods=['POST'])
def text2img():
    data = request.get_json(force=True)
    prompt = (data.get('prompt') or "").strip()
    if not prompt:
        return jsonify(error="Missing prompt"), 400

    image = pipe(prompt).images[0]
    fname = f"{uuid.uuid4().hex}.png"
    path = os.path.join(SAVE_DIR, fname)
    image.save(path)
    return jsonify(output=fname), 200

@app.route('/api/assets', methods=['GET'])
def assets():
    try:
        files = os.listdir(SAVE_DIR)
    except FileNotFoundError:
        files = []
    allowed = ('.png', '.jpg', '.jpeg', '.mp4')
    files = [f for f in files if f.lower().endswith(allowed)]
    return jsonify(files), 200

@app.route('/api/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(SAVE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)