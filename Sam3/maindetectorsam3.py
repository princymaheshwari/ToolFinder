import modal
import io
import base64

def get_models():
    import os
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    from transformers import Sam3Processor, Sam3Model
    Sam3Processor.from_pretrained("facebook/sam3")
    Sam3Model.from_pretrained("facebook/sam3")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .run_commands("pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
    .pip_install("transformers", "opencv-python-headless", "Pillow", "numpy", "huggingface_hub")
    .run_function(get_models, secrets=[modal.Secret.from_name("huggingface")])
)

app = modal.App("detect-tools-sam3", image=image)

@app.cls(gpu="H100")
class DetectTools:

    @modal.enter()
    def load(self):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        from transformers import Sam3Processor, Sam3Model

        self.device = "cuda"
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)

    @modal.method()
    def detect(self, tools, img_byt) -> dict:
        import torch
        import numpy as np
        import cv2
        from PIL import Image

        img = Image.open(io.BytesIO(img_byt)).convert("RGB")
        width, height = img.size

        canvas = np.array(img)
        green = np.array([0, 255, 0], dtype=np.uint8)
        detections = []

        # query each tool separately to avoid label merging
        for tool_name in tools:
            inputs = self.processor(images=img, text=tool_name, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.35,
                mask_threshold=0.35,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks = results["masks"]
            scores = results["scores"]

            if masks is None or len(masks) == 0:
                continue

            masks_np = masks.cpu().numpy()
            scores_np = scores.cpu().numpy()

            for i in range(len(masks_np)):
                mask = masks_np[i].astype(bool)
                score = float(scores_np[i])

                # green overlay
                canvas[mask] = (canvas[mask] * 0.55 + green * 0.45).astype(np.uint8)

                # contour edge
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8) * 255,
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

                # label
                ys, xs = np.where(mask)
                tx, ty = (int(xs.min()), int(ys.min()) - 8) if len(xs) > 0 else (10, 10)
                cv2.putText(
                    canvas, f"{tool_name} {score:.0%}",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                detections.append({"label": tool_name, "score": score})

        if not detections:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return {"image": base64.b64encode(buf.getvalue()).decode(), "detections": [], "count": 0}

        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        return {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "detections": detections,
            "count": len(detections),
        }