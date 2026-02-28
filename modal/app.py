import modal

app = modal.App("workshop-finder")

MODELS_VOL = modal.Volume.from_name("workshop-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "opencv-python-headless",
        "open-clip-torch",
        "transformers",
    )
)

@app.function(
    image=image,
    gpu="L4",
    volumes={"/models": MODELS_VOL},
    min_containers=1,
    max_containers=1,
)
@modal.fastapi_endpoint(method="POST")
def infer(payload: dict):
    return {"status": "gpu endpoint wired", "keys": list(payload.keys())}