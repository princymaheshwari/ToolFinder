import sys
import base64
from pathlib import Path
import modal

prompt = sys.argv[1]
tools = [t.strip() for t in prompt.split(".") if t.strip()]
img_path = sys.argv[2]

model = modal.Cls.from_name("detect-tools-yolo", "DetectTools")()

result = model.detect.remote(tools, Path(img_path).read_bytes())
Path("result.png").write_bytes(base64.b64decode(result["image"]))
print(f"Found {result['count']} tools:")
for d in result["detections"]:
    print(f"  {d['label']} ({d['score']:.0%})")
print("Saved → result.png")