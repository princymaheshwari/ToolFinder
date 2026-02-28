import torch
from PIL import Image
import open_clip

class VisionPipeline:

    def __init__(self, device="cuda"):
        self.device = device

        # Load CLIP only for now
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.clip_model = self.clip_model.to(self.device).eval()

    @torch.inference_mode()
    def detect_and_rank(self, image: Image.Image, query: str):

        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_tensor = self.clip_tokenizer([query]).to(self.device)

        image_features = self.clip_model.encode_image(image_tensor)
        text_features = self.clip_model.encode_text(text_tensor)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()

        return {
            "clip_similarity": similarity
        }