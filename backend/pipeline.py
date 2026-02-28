import torch
import numpy as np
import cv2
from PIL import Image
import open_clip
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor


class VisionPipeline:

    def __init__(self, device="cuda"):
        self.device = device

        # --- Load Grounding DINO ---
        self.gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(self.device)

        # --- Load SAM ---
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-large").to(self.device)

        # --- Load CLIP ---
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_model = self.clip_model.to(self.device).eval()

    @torch.inference_mode()
    def detect_and_rank(self, image: Image.Image, query: str):

        width, height = image.size

        # ---------------------
        # 1️⃣ DINO Detection
        # ---------------------
        dino_input = self.gd_processor(
            images=image,
            text=[query],
            return_tensors="pt"
        ).to(self.device)

        dino_output = self.gd_model(**dino_input)

        results = self.gd_processor.post_process_grounded_object_detection(
            dino_output,
            threshold=0.3,
            target_sizes=[(height, width)],
            text_labels=[query],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        if len(boxes) == 0:
            return {"best_box": None, "clip_score": 0.0}

        # ---------------------
        # 2️⃣ SAM Masking
        # ---------------------
        sam_inputs = self.sam_processor(
            images=image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt"
        ).to(self.device)

        sam_output = self.sam_model(**sam_inputs)

        masks_tensor = self.sam_processor.image_processor.post_process_masks(
            sam_output.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]

        best_idx = sam_output.iou_scores.cpu().numpy()[0].argmax(axis=1)

        # ---------------------
        # 3️⃣ CLIP Re-ranking
        # ---------------------
        crops = []

        for i in range(len(boxes)):
            mask = masks_tensor[i, best_idx[i]].numpy().astype(bool)

            masked_img = np.array(image)
            masked_img[~mask] = 0
            crops.append(Image.fromarray(masked_img))

        image_tensors = torch.stack([
            self.clip_preprocess(c).to(self.device) for c in crops
        ])

        text_tensor = self.clip_tokenizer([query]).to(self.device)

        image_features = self.clip_model.encode_image(image_tensors)
        text_features = self.clip_model.encode_text(text_tensor)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(1)

        best_index = similarities.argmax().item()

        return {
            "best_box": boxes[best_index].tolist(),
            "clip_score": float(similarities[best_index].item()),
            "num_candidates": len(boxes),
        }