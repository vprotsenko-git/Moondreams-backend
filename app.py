import os
import uuid
import threading
import time
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

# Global state for job progress (for demo, only one job at a time supported)
job_state = {
    "status": "idle",         # "idle", "running", "done", "error"
    "progress": 0,            # 0-100
    "output": None,           # filename if done
    "error": None
}
job_lock = threading.Lock()

def generate_image(prompt):
    global job_state
    try:
        with job_lock:
            job_state["status"] = "running"
            job_state["progress"] = 0
            job_state["output"] = None
            job_state["error"] = None

        # Simulate progress: 10 steps, update progress every 0.5s
        steps = 10
        for i in range(steps):
            time.sleep(0.5)
            with job_lock:
                job_state["progress"] = int((i + 1) / steps * 80)  # up to 80%

        # Actual generation
        result = pipe(prompt)
        image = result.images[0]
        fname = f"{uuid.uuid4().hex}.png"
        path = os.path.join(SAVE_DIR, fname)
        image.save(path)

        # Final progress
        with job_lock:
            job_state["progress"] = 100
            job_state["output"] = fname
            job_state["status"] = "done"
            job_state["error"] = None

        del image, result
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        with job_lock:
            job_state["status"] = "error"
            job_state["error"] = str(e)
            job_state["progress"] = 100

@app.route('/api/text2img', methods=['POST'])
def text2img():
    global job_state
    data = request.get_json(force=True) or {}
    prompt = data.get('prompt', '').strip()

    with job_lock:
        if job_state["status"] == "running":
            return jsonify(error="Generation already in progress"), 429
        # Reset state and start new thread
        job_state = {
            "status": "running",
            "progress": 0,
            "output": None,
            "error": None
        }
    t = threading.Thread(target=generate_image, args=(prompt,))
    t.start()
    return jsonify(status="started"), 202

@app.route('/api/text2img/status', methods=['GET'])
def text2img_status():
    with job_lock:
        return jsonify(
            status=job_state["status"],
            progress=job_state["progress"],
            output=job_state["output"],
            error=job_state["error"]
        )

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
    app.run(host='0.0.0.0', port=5000, debug=False)