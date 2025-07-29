import os
import uuid
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from diffusers import StableDiffusionPipeline
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
import torch

# Init
CORS(app := Flask(__name__))
app.secret_key = "gH7kLm9Pq2Rz5SvT"

SAVE_DIR = "/app/models"
USER_DB = "/app/users.json"
os.makedirs(SAVE_DIR, exist_ok=True)
if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to(device)


@app.route("/text2img", methods=["POST"])
def text2img():
    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    image = pipe(prompt).images[0]
    filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(SAVE_DIR, filename)
    image.save(output_path)

    return jsonify({"output": filename})


@app.route("/img2video", methods=["POST"])
def img2video():
    frames = request.json.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return jsonify({"error": "Missing frames"}), 400

    paths = [os.path.join(SAVE_DIR, f) for f in frames if f.endswith(".png")]
    images = [Image.open(p) for p in paths]
    clip = ImageSequenceClip([i for i in images], fps=1)
    filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(SAVE_DIR, filename)
    clip.write_videofile(output_path, codec="libx264")

    return jsonify({"output": filename})


@app.route("/models/<path:filename>")
def serve_file(filename):
    return app.send_from_directory(SAVE_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)