import os
import uuid
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageSequenceClip
from PIL import Image
import torch
import bcrypt

# Init
app = Flask(__name__)
CORS(app)

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret')
jwt = JWTManager(app)

SAVE_DIR = "/app/models"
USER_DB = "/app/users.json"
os.makedirs(SAVE_DIR, exist_ok=True)
if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Helpers
def load_users():
    with open(USER_DB) as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=2)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(force=True)
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"error": "Логін і пароль обов’язкові"}), 400
    # Логіка реєстрації (унікальність, хешування, запис)
    # ...
    return jsonify({"message": "Користувача успішно створено"}), 201

@app.route("/login", methods=["POST"])
def login():
    email = request.json.get("email")
    password = request.json.get("password")
    users = load_users()
    if email not in users:
        return jsonify({"msg": "Invalid email"}), 401
    if not bcrypt.checkpw(password.encode(), users[email]["password"].encode()):
        return jsonify({"msg": "Invalid password"}), 401
    token = create_access_token(identity=email)
    return jsonify({"access_token": token})

# Text to Image
@app.route("/text2img", methods=["POST"])
@jwt_required()
def text2img():
    user = get_jwt_identity()
    prompt = request.json.get("prompt")
    image = pipe(prompt).images[0]
    fname = f"{uuid.uuid4()}.png"
    path = os.path.join(SAVE_DIR, fname)
    image.save(path)
    return jsonify({"user": user, "output": fname})

# Image to Video
@app.route("/img2video", methods=["POST"])
@jwt_required()
def img2video():
    user = get_jwt_identity()
    frames = request.json.get("frames", [])
    clip = ImageSequenceClip([os.path.join(SAVE_DIR, f) for f in frames], fps=5)
    vid_name = f"{uuid.uuid4()}.mp4"
    vid_path = os.path.join(SAVE_DIR, vid_name)
    clip.write_videofile(vid_path, codec="libx264")
    return jsonify({"user": user, "output": vid_name})

# Upscale
@app.route("/upscale", methods=["POST"])
@jwt_required()
def upscale():
    user = get_jwt_identity()
    image_fname = request.json.get("image")
    img = Image.open(os.path.join(SAVE_DIR, image_fname))
    new = img.resize((img.width * 2, img.height * 2), resample=Image.LANCZOS)
    out_name = f"upscaled-{image_fname}"
    out_path = os.path.join(SAVE_DIR, out_name)
    new.save(out_path)
    return jsonify({"user": user, "output": out_name})

# Train placeholder
@app.route("/train-model", methods=["POST"])
@jwt_required()
def train_model():
    user = get_jwt_identity()
    return jsonify({"user": user, "status": "training started (placeholder)"}), 202

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)