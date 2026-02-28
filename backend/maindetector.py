import modal
import io
import base64

# ---------------------------------------------------------------------------
# Pre-download all model weights into the container image at build time.
# This means the first real request is fast — models are already on disk.
# ---------------------------------------------------------------------------
def get_models():
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        SamModel,
        SamProcessor,
    )
    import open_clip

    AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    SamProcessor.from_pretrained("facebook/sam-vit-large")
    SamModel.from_pretrained("facebook/sam-vit-large")
    open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers>=4.40.0",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "open-clip-torch",
    )
    .run_function(get_models)
)

app = modal.App("detect-tools-clip", image=image)

# Discard any detection CLIP scores below this — almost certainly a false positive
CLIP_THRESHOLD = 0.18
# Maximum number of DINO boxes forwarded to SAM (keeps inference fast)
MAX_CANDIDATES = 6


@app.cls(gpu="A10G")
class DetectTools:

    # -----------------------------------------------------------------------
    # Runs once when the container starts. All three models live in GPU memory
    # for the lifetime of the container — zero reload cost per request.
    # -----------------------------------------------------------------------
    @modal.enter()
    def load(self):
        import torch
        import open_clip
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
            SamModel,
            SamProcessor,
        )

        self.device = "cuda"

        # Grounding DINO — open-vocabulary bounding box detection
        self.gd_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        )
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(self.device)

        # SAM — pixel-precise mask from bounding box prompt
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-large").to(
            self.device
        )

        # CLIP — semantic re-ranking: scores (masked crop, label text) pairs
        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_model = self.clip_model.to(self.device).eval()

    # -----------------------------------------------------------------------
    # Main inference method.
    #
    # Pipeline:
    #   1. DINO   → candidate bounding boxes + coarse labels
    #   2. SAM    → precise pixel masks for each box
    #   3. CLIP   → re-rank: score each (bbox-cropped masked region, label) pair
    #   4. Draw   → semi-transparent green fill + contour + "label CLIP_score%"
    #
    # Args:
    #   tools     – list of strings, e.g. ["hammer", "pliers"]
    #   img_bytes – raw image bytes (JPEG / PNG)
    #
    # Returns dict:
    #   image      – base64-encoded PNG with detections drawn
    #   detections – list of {label, dino_score, clip_score}
    #   count      – number of accepted detections
    # -----------------------------------------------------------------------
    @modal.method()
    def detect(self, tools: list, img_bytes: bytes) -> dict:
        import torch
        import numpy as np
        import cv2
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = img.size

        # ── 1. Grounding DINO ───────────────────────────────────────────────
        # text=[tools] wraps the label list in a batch-of-1 outer list, which
        # is what the DINO processor expects for a single image.
        dino_input = self.gd_processor(
            images=img, text=[tools], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            dino_output = self.gd_model(**dino_input)

        results = self.gd_processor.post_process_grounded_object_detection(
            dino_output,
            threshold=0.3,
            target_sizes=[(height, width)],
            text_labels=[tools],
        )[0]

        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["text_labels"]

        # Drop any box whose decoded label is not in the requested tools list
        valid  = [i for i, l in enumerate(labels) if l in tools]
        boxes  = boxes[valid]
        scores = scores[valid]
        labels = [labels[i] for i in valid]

        if len(boxes) == 0:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return {
                "image":      base64.b64encode(buf.getvalue()).decode(),
                "detections": [],
                "count":      0,
            }

        # Keep only the top-N by DINO score before running the heavier SAM
        if len(boxes) > MAX_CANDIDATES:
            top    = np.argsort(scores)[::-1][:MAX_CANDIDATES]
            boxes  = boxes[top]
            scores = scores[top]
            labels = [labels[i] for i in top]

        # ── 2. SAM segmentation ─────────────────────────────────────────────
        sam_inputs = self.sam_processor(
            images=img, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sam_output = self.sam_model(**sam_inputs)

        # Upscale masks back to original image resolution
        masks_tensor = self.sam_processor.image_processor.post_process_masks(
            sam_output.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]

        # SAM returns 3 mask candidates per box; pick the one with highest IoU
        if masks_tensor.ndim == 3:          # single-box edge-case guard
            masks_tensor = masks_tensor.unsqueeze(0)
        best_mask_idx = sam_output.iou_scores.cpu().numpy()[0].argmax(axis=1)

        # ── 3. CLIP re-ranking ──────────────────────────────────────────────
        # For each detection:
        #   a. Zero out background pixels using the SAM mask
        #   b. Crop tightly to the bounding box  ← key fix vs. full-frame CLIP
        #   c. Score (crop, label_text) with CLIP
        #
        # Cropping to the bbox before CLIP preprocessing is critical: without
        # it CLIP's center-crop discards the object and scores collapse to ~0.23.
        crops = []
        for i in range(len(boxes)):
            mask = masks_tensor[i, best_mask_idx[i]].numpy().astype(bool)
            x1 = max(0,      int(boxes[i][0]))
            y1 = max(0,      int(boxes[i][1]))
            x2 = min(width,  int(boxes[i][2]))
            y2 = min(height, int(boxes[i][3]))
            arr = np.array(img).copy()
            arr[~mask] = 0                          # zero out background
            crops.append(Image.fromarray(arr[y1:y2, x1:x2]))

        with torch.no_grad():
            img_tensors = torch.stack(
                [self.clip_preprocess(c).to(self.device) for c in crops]
            )
            img_feats = self.clip_model.encode_image(img_tensors)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # Score each crop against its own label independently.
            # Batching all labels as a single text-matrix would mix signals when
            # multiple different tools are requested at once.
            clip_scores = []
            for i, label in enumerate(labels):
                tok       = self.clip_tokenizer([label]).to(self.device)
                txt_feat  = self.clip_model.encode_text(tok)
                txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                clip_scores.append((img_feats[i : i + 1] @ txt_feat.T).item())

        clip_scores = np.array(clip_scores)

        # ── 4. Visualise ────────────────────────────────────────────────────
        canvas = np.array(img)
        green  = np.array([0, 255, 0], dtype=np.uint8)
        detections = []

        for i in range(len(boxes)):
            # Filter: if CLIP is very uncertain, DINO was likely a false positive
            if clip_scores[i] < CLIP_THRESHOLD:
                continue

            mask = masks_tensor[i, best_mask_idx[i]].numpy().astype(bool)

            # Semi-transparent green fill (45 % green, 55 % original)
            canvas[mask] = (canvas[mask] * 0.55 + green * 0.45).astype(np.uint8)

            # Sharp contour outline
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

            # Label: anchored to the top-left corner of the mask, score = CLIP
            ys, xs = np.where(mask)
            tx = int(xs.min()) if len(xs) > 0 else 10
            ty = max(12, int(ys.min()) - 8) if len(ys) > 0 else 10
            cv2.putText(
                canvas,
                f"{labels[i]} {clip_scores[i]:.0%}",
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            detections.append(
                {
                    "label":      labels[i],
                    "dino_score": float(scores[i]),
                    "clip_score": float(clip_scores[i]),
                }
            )

        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        return {
            "image":      base64.b64encode(buf.getvalue()).decode(),
            "detections": detections,
            "count":      len(detections),
        }
