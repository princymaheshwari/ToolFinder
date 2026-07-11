import modal
import io
import base64

def get_models():
    import os
    from inference import get_model
    get_model(model_id="yolotrainingdatasethackillinois/2")

    from transformers import SamModel, SamProcessor
    SamProcessor.from_pretrained("facebook/sam-vit-large")
    SamModel.from_pretrained("facebook/sam-vit-large")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1",
        "transformers>=4.40.0", "Pillow", "numpy",
        "opencv-python-headless", "inference",
    )
    .run_function(get_models, secrets=[modal.Secret.from_name("roboflow")])
)

app = modal.App("detect-tools-yolo", image=image)

@app.cls(gpu="A10G", secrets=[modal.Secret.from_name("roboflow")])
class DetectTools:

    @modal.enter()
    def load(self):
        import torch
        from inference import get_model
        from transformers import SamModel, SamProcessor

        self.device = "cuda"

        self.yolo = get_model(model_id="yolotrainingdatasethackillinois/2")

        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-large").to(self.device)

    @modal.method()
    def detect(self, tools, img_byt) -> dict:
        import torch
        import numpy as np
        import cv2
        from PIL import Image

        img = Image.open(io.BytesIO(img_byt)).convert("RGB")
        img_np = np.array(img)

        # YOLO inference
        results = self.yolo.infer(img_np, confidence=0.1)[0]

        bounding_boxes = []
        scores = []
        labels = []

        for pred in results.predictions:
            x1 = pred.x - pred.width / 2
            y1 = pred.y - pred.height / 2
            x2 = pred.x + pred.width / 2
            y2 = pred.y + pred.height / 2
            bounding_boxes.append([x1, y1, x2, y2])
            scores.append(pred.confidence)
            labels.append(pred.class_name)

        # filter to only requested tools
        tools_lower = [t.lower() for t in tools]
        print("YOLO raw detections:", [(l, s) for l, s in zip(labels, scores)])
        print("Looking for tools:", tools_lower)
        filtered = [
            (bb, s, l) for bb, s, l in zip(bounding_boxes, scores, labels)
            if l.lower() in tools_lower
        ]

        if not filtered:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return {"image": base64.b64encode(buf.getvalue()).decode(), "detections": [], "count": 0}

        bounding_boxes = np.array([f[0] for f in filtered])
        scores = np.array([f[1] for f in filtered])
        labels = [f[2] for f in filtered]

        # SAM segmentation from YOLO boxes
        sam_inputs = self.sam_processor(
            images=img, input_boxes=[bounding_boxes.tolist()], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sam_output = self.sam_model(**sam_inputs)

        masks_tensor = self.sam_processor.image_processor.post_process_masks(
            sam_output.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]

        best_idx = sam_output.iou_scores.cpu().numpy()[0].argmax(axis=1)

        # draw green overlays
        canvas = img_np.copy()
        green = np.array([0, 255, 0], dtype=np.uint8)
        detections = []

        for i in range(len(bounding_boxes)):
            mask = masks_tensor[i, best_idx[i]].numpy().astype(bool)

            canvas[mask] = (canvas[mask] * 0.55 + green * 0.45).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

            ys, xs = np.where(mask)
            tx, ty = (int(xs.min()), int(ys.min()) - 8) if len(xs) > 0 else (10, 10)
            cv2.putText(
                canvas, f"{labels[i]} {scores[i]:.0%}",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            detections.append({"label": labels[i], "score": float(scores[i])})

        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        return {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "detections": detections,
            "count": len(detections),
        }