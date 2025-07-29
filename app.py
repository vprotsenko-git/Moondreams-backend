import os, uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)
CORS(app)

SAVE = "models"
os.makedirs(SAVE, exist_ok=True)

# init model
device = "cuda" if os.getenv("USE_CUDA")=="1" else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

@app.route("/text2img", methods=["POST"])
def t2i():
    p = request.json.get("prompt","").strip()
    if not p: return jsonify(error="Missing prompt"), 400
    img = pipe(p).images[0]
    name = f"{uuid.uuid4().hex}.png"
    path = os.path.join(SAVE, name)
    img.save(path)
    return jsonify(output=name)

@app.route("/models/<path:f>")
def serve(f):
    return send_from_directory(SAVE, f)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)