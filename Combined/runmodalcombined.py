import sys
import base64
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import modal

# Minimum confidence required before a YOLO result is trusted for labels that
# have a SAM3 fallback. ~0.45 is a reasonable "good enough" bar.
FALLBACK_THRESHOLD = 0.45

def route(prompt: str, img_bytes: bytes, yolo, sam3, color: tuple = (0, 255, 0)) -> dict:
    

    if prompt == "Drill Bits":
        return yolo.detect.remote(["Drill Bits"], img_bytes, color=color)

    if prompt == "Camera":
        return yolo.detect.remote(["Camera"], img_bytes, min_confidence=0.25, color=color)

    if prompt == "Pins":
        return yolo.detect.remote(["Pins"], img_bytes, color=color)

    if prompt == "ESP32":
        return yolo.detect.remote(["ESP32"], img_bytes, color=color)

    if prompt == "Screwdriver":
        return yolo.detect.remote(["Screwdriver"], img_bytes, color=color)

    if prompt == "Allen key":
        return yolo.detect.remote(["Allen key"], img_bytes, color=color)

    if prompt == "Brush":
        return yolo.detect.remote(["Brush"], img_bytes, color=color)

    if prompt == "Screwdriver Kit":
        return sam3.detect.remote(["green case"], img_bytes, display_label="Screwdriver Kit", color=color)

    if prompt == "Motor":
        return yolo.detect.remote(["Motor"], img_bytes, color=color)

    if prompt == "Clutter":
        yolo_result = yolo.detect.remote(["Clutter"], img_bytes, max_detections=3, color=color)
        remaining = 3 - yolo_result["count"]

        if remaining <= 0:
            return yolo_result

        clutter_words = ["bottle", "food", "wrapper", "can", "cup", "box", "tissue", "snack bag", "trash"]
        yolo_img_bytes = base64.b64decode(yolo_result["image"])
        sam3_result = sam3.detect.remote(
            clutter_words, img_bytes,
            canvas_image_bytes=yolo_img_bytes,
            display_label="Clutter",
            max_detections=remaining,
            color=color,
        )

        combined = yolo_result["detections"] + sam3_result["detections"]
        return {
            "image": sam3_result["image"],
            "detections": combined,
            "count": len(combined),
        }

    if prompt == "Slider":
        result = yolo.detect.remote(["Slider"], img_bytes, min_confidence=FALLBACK_THRESHOLD, color=color)
        if result["count"] == 0:
            result = sam3.detect.remote(["Metallic Drawer Slider"], img_bytes, display_label="Slider", color=color)
        return result

    if prompt == "Tape":
        result = yolo.detect.remote(["Tape"], img_bytes, min_confidence=FALLBACK_THRESHOLD, color=color)
        if result["count"] == 0:
            result = sam3.detect.remote(["red tape holder"], img_bytes, display_label="Tape", color=color)
        return result

    if prompt == "Motor Controllers":
        result = yolo.detect.remote(
            ["Motor Controllers"], img_bytes,
            max_detections=2, min_confidence=FALLBACK_THRESHOLD, color=color,
        )
        if result["count"] == 0:
            result = sam3.detect.remote(["red plastic bags"], img_bytes, max_detections=2, display_label="Motor Controllers", color=color)
        return result

    return sam3.detect.remote([prompt], img_bytes, color=color)


def main():
    # Usage: python runmodalcombined.py <image_path> <prompt1> [prompt2 ...]
    # e.g.   python runmodalcombined.py test4.jpg Clutter Camera "Motor Controllers"
    if len(sys.argv) < 3:
        print("Usage: python runmodalcombined.py <image_path> <prompt1> [prompt2 ...]")
        sys.exit(1)

    img_path  = sys.argv[1]
    prompts   = [p.strip() for p in sys.argv[2:]]
    img_bytes = Path(img_path).read_bytes()

    yolo = modal.Cls.from_name("detect-tools-combined", "YoloSam2Detector")()
    sam3 = modal.Cls.from_name("detect-tools-combined", "Sam3Detector")()

    print(f"Image: {img_path} | Prompts: {prompts}")

    def detect(prompt):
        return prompt, route(prompt, img_bytes, yolo, sam3)

    # Run all prompts in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        for prompt, result in pool.map(detect, prompts):
            results[prompt] = result

    # Composite all overlays onto one image
    original  = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    composite = original.copy()

    for prompt in prompts:
        result    = results[prompt]
        result_np = np.array(Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB"))
        diff_mask = np.any(result_np != original, axis=2)
        composite[diff_mask] = result_np[diff_mask]

        count = result["count"]
        print(f"\n'{prompt}' → {count} detection(s):")
        for d in result["detections"]:
            print(f"  {d['label']} ({d['score']:.0%})")
        if count == 1:
            print(f"  Position: ({result['cx']}, {result['cy']})")

    buf = io.BytesIO()
    Image.fromarray(composite).save(buf, format="PNG")
    Path("result.png").write_bytes(buf.getvalue())
    print("\nSaved → result.png")


if __name__ == "__main__":
    main()
