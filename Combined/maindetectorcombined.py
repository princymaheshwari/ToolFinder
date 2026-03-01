import modal
import io
import base64

# ── model pre-download functions ──────────────────────────────────────────────

def get_yolo_models():
    import os
    from inference import get_model
    get_model(model_id="yolotrainingdatasethackillinois/2")
    from transformers import SamModel, SamProcessor
    SamProcessor.from_pretrained("facebook/sam-vit-large")
    SamModel.from_pretrained("facebook/sam-vit-large")

def get_sam3_models():
    import os
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    from transformers import Sam3Processor, Sam3Model
    Sam3Processor.from_pretrained("facebook/sam3")
    Sam3Model.from_pretrained("facebook/sam3")

# ── container images ──────────────────────────────────────────────────────────

yolo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1",
        "transformers>=4.40.0", "Pillow", "numpy",
        "opencv-python-headless", "inference",
    )
    .run_function(get_yolo_models, secrets=[modal.Secret.from_name("roboflow")])
)

sam3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .run_commands("pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
    .pip_install("transformers", "opencv-python-headless", "Pillow", "numpy", "huggingface_hub")
    .run_function(get_sam3_models, secrets=[modal.Secret.from_name("huggingface")])
)

app = modal.App("detect-tools-combined")


@app.cls(gpu="A10G", image=yolo_image, secrets=[modal.Secret.from_name("roboflow")])
class YoloSam2Detector:

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
    def detect(self, tools: list, img_byt: bytes, max_detections: int = None, min_confidence: float = 0.1, color: tuple = (0, 255, 0)) -> dict:
        import torch
        import numpy as np
        import cv2
        from PIL import Image

        img = Image.open(io.BytesIO(img_byt)).convert("RGB")
        img_np = np.array(img)
        results = self.yolo.infer(img_np, confidence=0.1)[0]

        bounding_boxes, scores, labels = [], [], []
        for pred in results.predictions:
            x1 = pred.x - pred.width / 2
            y1 = pred.y - pred.height / 2
            x2 = pred.x + pred.width / 2
            y2 = pred.y + pred.height / 2
            bounding_boxes.append([x1, y1, x2, y2])
            scores.append(pred.confidence)
            labels.append(pred.class_name)

        tools_lower = [t.lower() for t in tools]
        print("YOLO raw detections:", [(l, f"{s:.0%}") for l, s in zip(labels, scores)])
        print("Looking for:", tools_lower, "| min_confidence:", min_confidence)

        filtered = [
            (bb, s, l) for bb, s, l in zip(bounding_boxes, scores, labels)
            if l.lower() in tools_lower and s >= min_confidence
        ]

        if not filtered:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return {"image": base64.b64encode(buf.getvalue()).decode(), "detections": [], "count": 0}

        # sort descending by confidence, then cap to max_detections
        filtered.sort(key=lambda x: x[1], reverse=True)
        if max_detections is not None:
            filtered = filtered[:max_detections]

        bounding_boxes = np.array([f[0] for f in filtered])
        scores_arr    = np.array([f[1] for f in filtered])
        labels        = [f[2] for f in filtered]

        # SAM segmentation using YOLO boxes as prompts
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

        canvas = img_np.copy()
        overlay_rgb = np.array(color, dtype=np.uint8)
        draw_bgr    = (color[2], color[1], color[0])   # cv2 draws in BGR on our RGB canvas
        detections  = []

        for i in range(len(bounding_boxes)):
            mask = masks_tensor[i, best_idx[i]].numpy().astype(bool)
            canvas[mask] = (canvas[mask] * 0.55 + overlay_rgb * 0.45).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, draw_bgr, 2)

            ys, xs = np.where(mask)
            cx = int(xs.mean()) if len(xs) > 0 else 0
            cy = int(ys.mean()) if len(ys) > 0 else 0
            tx, ty = (int(xs.min()), int(ys.min()) - 12) if len(xs) > 0 else (10, 10)
            cv2.putText(
                canvas, f"{labels[i]} {scores_arr[i]:.0%}",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.8, draw_bgr, 3
            )
            detections.append({"label": labels[i], "score": float(scores_arr[i]), "cx": cx, "cy": cy})

        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        result = {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "detections": detections,
            "count": len(detections),
        }
        if len(detections) == 1:
            result["cx"] = detections[0]["cx"]
            result["cy"] = detections[0]["cy"]
        return result

# ── SAM3 detector ─────────────────────────────────────────────────────────────

@app.cls(gpu="H100", image=sam3_image, secrets=[modal.Secret.from_name("huggingface")])
class Sam3Detector:

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
    def detect(self, tools: list, img_byt: bytes, max_detections: int = None, threshold: float = 0.35, canvas_image_bytes: bytes = None, display_label: str = None, color: tuple = (0, 255, 0)) -> dict:
        import torch
        import numpy as np
        import cv2
        from PIL import Image

        # img is used for model inference; canvas is what we draw overlays onto.
        # Passing canvas_image_bytes lets callers supply a pre-rendered base image
        # (e.g. YOLO output) so SAM3 overlays are composited on top of it.
        img = Image.open(io.BytesIO(img_byt)).convert("RGB")

        all_detections = []  # list of {"label", "score", "mask"}

        for tool_name in tools:
            inputs = self.processor(images=img, text=tool_name, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks  = results["masks"]
            scores = results["scores"]

            if masks is None or len(masks) == 0:
                continue

            masks_np  = masks.cpu().numpy()
            scores_np = scores.cpu().numpy()

            for i in range(len(masks_np)):
                all_detections.append({
                    "label": tool_name,
                    "score": float(scores_np[i]),
                    "mask":  masks_np[i].astype(bool),
                })

        if not all_detections:
            base = Image.open(io.BytesIO(canvas_image_bytes)).convert("RGB") if canvas_image_bytes else img
            buf = io.BytesIO()
            base.save(buf, format="PNG")
            return {"image": base64.b64encode(buf.getvalue()).decode(), "detections": [], "count": 0}

        # sort descending by confidence, then cap
        all_detections.sort(key=lambda x: x["score"], reverse=True)
        if max_detections is not None:
            all_detections = all_detections[:max_detections]

        # draw on the provided canvas (e.g. YOLO output) or fall back to the original
        if canvas_image_bytes:
            canvas = np.array(Image.open(io.BytesIO(canvas_image_bytes)).convert("RGB"))
        else:
            canvas = np.array(img)
        overlay_rgb = np.array(color, dtype=np.uint8)
        draw_bgr    = (color[2], color[1], color[0])   # cv2 draws in BGR on our RGB canvas

        for det in all_detections:
            mask = det["mask"]
            canvas[mask] = (canvas[mask] * 0.55 + overlay_rgb * 0.45).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, draw_bgr, 2)

            text = display_label if display_label else det["label"]
            ys, xs = np.where(mask)
            det["cx"] = int(xs.mean()) if len(xs) > 0 else 0
            det["cy"] = int(ys.mean()) if len(ys) > 0 else 0
            tx, ty = (int(xs.min()), int(ys.min()) - 12) if len(xs) > 0 else (10, 10)
            cv2.putText(
                canvas, f"{text} {det['score']:.0%}",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.8, draw_bgr, 3
            )

        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        out_label = display_label if display_label else None
        out_detections = [
            {"label": out_label or d["label"], "score": d["score"], "cx": d["cx"], "cy": d["cy"]}
            for d in all_detections
        ]
        result = {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "detections": out_detections,
            "count": len(out_detections),
        }
        if len(out_detections) == 1:
            result["cx"] = out_detections[0]["cx"]
            result["cy"] = out_detections[0]["cy"]
        return result
