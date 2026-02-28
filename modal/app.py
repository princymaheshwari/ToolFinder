import base64
import io
from PIL import Image
import torch
import modal

from backend.pipeline import VisionPipeline

app = modal.App("workshop-finder")

MODELS_VOL = modal.Volume.from_name("workshop-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers>=4.40.0",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "open-clip-torch",
    )
    .add_local_dir("backend", remote_path="/root/backend")
)

def decode_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

@app.cls(
    image=image,
    gpu="L4",
    volumes={"/models": MODELS_VOL},
    min_containers=1,
    max_containers=1,
)
class Pipeline:

    @modal.enter()
    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = VisionPipeline(device=self.device)

    @modal.fastapi_endpoint(method="POST")
    def infer(self, payload: dict):

        pil_img = decode_image(payload["image_b64"])
        query = payload["query"]

        result = self.pipeline.detect_and_rank(pil_img, query)

        return result