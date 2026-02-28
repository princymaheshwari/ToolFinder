import modal
import io
import base64

#load dino+sam once
def get_models():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
    AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    SamProcessor.from_pretrained("facebook/sam-vit-large")
    SamModel.from_pretrained("facebook/sam-vit-large")

# generic setup modal for image container
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", "transformers>=4.40.0", "Pillow", "numpy", "opencv-python-headless")
    .run_function(get_models)
)

app = modal.App("detect-tools", image=image)

# .cls for resuable class
@app.cls(gpu="A10G")
class DetectTools:

    # will run on container start, so load everything here
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

        self.device = "cuda"

        #set up grounding dino
        self.gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)

        # set up sam
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-large").to(self.device)

    # main detection method
    @modal.method()
    def detect(self, tools, img_byt) -> dict:
        import torch
        import numpy as np
        import cv2
        from PIL import Image

        # have to reconstruct a PIL image using bytes to send to modal
        img = Image.open(io.BytesIO(img_byt)).convert("RGB")
        width, height = img.size

        # main grounding DINO logic

        # convert image and prompt inputs into model arguments
        dino_input = self.gd_processor(images=img, text=[tools], return_tensors="pt").to(self.device)
       
        with torch.no_grad():
            # unpack dict
            dino_output = self.gd_model(**dino_input)
        

        # filter by acceptable thresholds and get pixel coordinates
        results = self.gd_processor.post_process_grounded_object_detection(
            dino_output,
            threshold=0.35,
            target_sizes=[(height, width)],
            text_labels=[tools],
        )[0]

        

        bounding_boxes = results["boxes"].cpu().numpy()
        # confidence scores
        scores = results["scores"].cpu().numpy()
        # labels to match with the prompt
        labels = results["text_labels"]

        valid = [i for i, l in enumerate(labels) if l in tools]
        bounding_boxes = bounding_boxes[valid]
        scores = scores[valid]
        labels = [labels[i] for i in valid]


        # nothing found
        if len(bounding_boxes) == 0:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return {"image": base64.b64encode(buf.getvalue()).decode(), "detections": [], "count": 0}
        # main sam logic

        #specific [[[x, x, y, y], ....]] format
        sam_inputs = self.sam_processor(images=img, input_boxes=[bounding_boxes.tolist()], return_tensors="pt").to(self.device)

        with torch.no_grad():
            sam_output = self.sam_model(**sam_inputs)
        

        # upscale to original img size
        masks_tensor = self.sam_processor.image_processor.post_process_masks(
            sam_output.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]

        # 3 masks each bounding box + score
        best_idx = sam_output.iou_scores.cpu().numpy()[0].argmax(axis=1)

        # basically draw a green overlay
        canvas = np.array(img)
        green = np.array([0, 255, 0], dtype=np.uint8)
        detections = []

        for i in range(len(bounding_boxes)):
            mask = masks_tensor[i, best_idx[i]].numpy().astype(bool) 

            
            canvas[mask] = (canvas[mask] * 0.55 + green * 0.45).astype(np.uint8)

            
            contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

            # Draw label text near the top of the mask
            ys, xs = np.where(mask)
            tx, ty = (int(xs.min()), int(ys.min()) - 8) if len(xs) > 0 else (10, 10)
            cv2.putText(canvas, f"{labels[i]} {scores[i]:.0%}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detections.append({"label": labels[i], "score": float(scores[i])})
        
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode(), "detections": detections, "count": len(detections)}




