import sys
import base64
from pathlib import Path
import modal

# ---------------------------------------------------------------------------
# Usage:
#   python runmodalsam3.py "hammer. pliers. wrench" image.jpg
#
# Deploy (run once, or after any change to backend/maindetectorsam3.py):
#   modal deploy backend/maindetectorsam3.py
# ---------------------------------------------------------------------------

if len(sys.argv) != 3:
    print('Usage: python runmodalsam3.py "tool1. tool2" image.jpg')
    sys.exit(1)

prompt   = sys.argv[1]
img_path = sys.argv[2]

tools = [t.strip() for t in prompt.split(".") if t.strip()]

print(f"Querying for: {tools}")
print(f"Image:        {img_path}")

model  = modal.Cls.from_name("detect-tools-sam3", "DetectTools")()
result = model.detect.remote(tools, Path(img_path).read_bytes())

Path("result.png").write_bytes(base64.b64decode(result["image"]))
print(f"\nSaved → result.png")
print(f"Found {result['count']} detection(s):\n")

for d in result["detections"]:
    bar = "█" * int(d["score"] * 20)
    print(f"  {d['label']:<20} {d['score']:.0%}  {bar}")
