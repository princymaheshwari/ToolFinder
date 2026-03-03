import modal

# 1. Configuration
VOLUME_NAME = "sam3-weights-cache"
SECRET_NAME = "huggingface-secret"
CACHE_PATH = "/my_vol"

# 2. Define Infrastructure
app = modal.App("sam3-segmentation")

image = (
  modal.Image.debian_slim(python_version="3.11")
  .apt_install("git")
  .pip_install(
    "torch",
    "torchvision",
    "transformers @ git+https://github.com/huggingface/transformers.git",
    "huggingface_hub",
    "pillow",
    "numpy",
    "matplotlib",
    "requests",
  )
)

# 3. Remote Segmentation Function
@app.function(
  gpu="a100",
  image=image,
  secrets=[modal.Secret.from_name(SECRET_NAME)],
  volumes={CACHE_PATH: modal.Volume.from_name(VOLUME_NAME)}
)
def segment_tool(image_bytes: bytes, prompt: str):
  import os
  import io
  import torch
  from PIL import Image
  from transformers import Sam3Processor, Sam3Model

  os.environ["HF_HOME"] = CACHE_PATH

  token = os.environ["HF_TOKEN"]

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"[SAM3] Loading model on {device}...")

  model   = Sam3Model.from_pretrained("facebook/sam3", token=token).to(device)
  processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)

  pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

  print(f"[SAM3] Running prompt: '{prompt}'")
  inputs = processor(
    images=pil_image,
    text=prompt,
    return_tensors="pt"
  ).to(device)

  with torch.no_grad():
    outputs = model(**inputs)

  results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
  )[0]

  print(f"[SAM3] Found {len(results['masks'])} object(s)")

  return {
    "masks":    results["masks"].cpu().numpy().tolist(),
    "boxes":    results["boxes"].cpu().numpy().tolist(),
    "scores":   results["scores"].cpu().numpy().tolist(),
    "image_size": list(pil_image.size),  # [W, H]
  }


# 4. Visualization (runs locally)
def visualize(image_path: str, result: dict, output_path: str, prompt: str):
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.patches as mpatches
  from PIL import Image

  img  = np.array(Image.open(image_path).convert("RGB"))
  masks  = np.array(result["masks"])  # (N, H, W) bool
  boxes  = result["boxes"]      # (N, 4) xyxy
  scores = result["scores"]
  n    = len(masks)

  fig, axes = plt.subplots(1, 2, figsize=(14, 7))
  fig.suptitle(f'SAM3  |  prompt: "{prompt}"', fontsize=13)

  axes[0].imshow(img)
  axes[0].set_title("Original Image")
  axes[0].axis("off")

  axes[1].imshow(img)
  axes[1].set_title(f"{n} instance(s) detected")
  axes[1].axis("off")

  cmap = plt.cm.get_cmap("tab10", max(n, 1))
  legend_handles = []

  for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
    color = np.array(cmap(i % 10))[:3]

    # Mask overlay
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask.astype(bool)] = [*color, 0.45]
    axes[1].imshow(overlay)
    axes[1].contour(mask, levels=[0.5], colors=[color], linewidths=1.5)

    # Bounding box
    x1, y1, x2, y2 = box
    rect = mpatches.Rectangle(
      (x1, y1), x2 - x1, y2 - y1,
      linewidth=2, edgecolor=color, facecolor="none"
    )
    axes[1].add_patch(rect)
    axes[1].text(x1, y1 - 4, f"{score:.2f}",
           color=color, fontsize=8, fontweight="bold")

    legend_handles.append(
      mpatches.Patch(color=color, label=f"Instance {i+1}  score={score:.2f}")
    )

  if legend_handles:
    axes[1].legend(handles=legend_handles, loc="upper right",
             fontsize=8, framealpha=0.7)

  plt.tight_layout()
  plt.savefig(output_path, dpi=150, bbox_inches="tight")
  plt.close()
  print(f"[viz] Saved → {output_path}")


# 5. Local Entrypoint
@app.local_entrypoint()
def main(
  image_path: str,
  prompt: str,
  output_path: str = "segmentation_result.png",
  show: bool = False,
):
  """
  Run SAM3 text-prompted segmentation remotely on Modal, visualise locally.

  Usage:
    modal run sam3_segment.py --image-path ./workbench.png --prompt "the hammer"
    modal run sam3_segment.py --image-path ./photo.jpg --prompt "red cup" --show
  """
  import json

  print(f"[main] Reading image: {image_path}")
  with open(image_path, "rb") as f:
    image_bytes = f.read()

  print(f"[main] Sending to SAM3  |  prompt='{prompt}'")
  result = segment_tool.remote(image_bytes, prompt)

  n = len(result["masks"])
  print(f"\n[main] Received {n} instance(s).")
  for i, score in enumerate(result["scores"]):
    print(f"     Instance {i+1}: score={score:.4f}  box={result['boxes'][i]}")

  if n > 0:
    visualize(image_path, result, output_path, prompt)
  else:
    print("[main] No instances detected — skipping visualisation.")

  if show and n > 0:
    import subprocess, sys
    opener = {"darwin": "open", "win32": "start", "linux": "xdg-open"}.get(
      sys.platform, "xdg-open"
    )
    subprocess.run([opener, output_path])

  print("\n[main] Summary:")
  print(json.dumps({
    "prompt":    prompt,
    "num_instances": n,
    "scores":    result["scores"],
    "boxes":     result["boxes"],
    "image_size":  result["image_size"],
    "output_path":   output_path,
  }, indent=2))