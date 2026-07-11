import sys
import base64
from pathlib import Path
import modal

# ---------------------------------------------------------------------------
# Usage:
#   python runmodal.py "hammer. pliers. wrench" image.jpg
#
# Tools are split on "." so you can pass one or many:
#   python runmodal.py "hammer" test.jpg
#   python runmodal.py "hammer. screwdriver. pliers" workshop.jpg
#
# Deploy (run once, or after any change to backend/maindetector.py):
#   modal deploy backend/maindetector.py
# ---------------------------------------------------------------------------

if len(sys.argv) != 3:
    print('Usage: python runmodal.py "tool1. tool2. tool3" image.jpg')
    sys.exit(1)

prompt   = sys.argv[1]
img_path = sys.argv[2]

tools = [t.strip() for t in prompt.split(".") if t.strip()]

print(f"Querying for: {tools}")
print(f"Image:        {img_path}")

model  = modal.Cls.from_name("detect-tools-clip", "DetectTools")()
result = model.detect.remote(tools, Path(img_path).read_bytes())

Path("result.png").write_bytes(base64.b64decode(result["image"]))
print(f"\nSaved → result.png")
print(f"Found {result['count']} detection(s):\n")

for d in result["detections"]:
    bar = "█" * int(d["clip_score"] * 20)
    print(f"  {d['label']:<20} CLIP {d['clip_score']:.0%}  {bar}")
    print(f"  {'':20} DINO {d['dino_score']:.0%}")
    print()
