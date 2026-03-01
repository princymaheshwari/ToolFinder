import io
import base64
import os
import sys
import socket
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import modal
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

from runmodalcombined import route

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PI_HOST    = "192.168.50.232"
PI_PORT    = 9999
DISPLAY_MS = 5000

# RGB colors cycled per prompt when multiple objects are requested.
# Single-object requests always use green (index 0).
PALETTE = [
    (0,   255,   0),   # green
    (0,   255, 255),   # cyan
    (255, 255,   0),   # yellow
    (255,   0, 255),   # magenta
    (255, 165,   0),   # orange
    (100, 100, 255),   # blue
]

# ---------------------------------------------------------------------------
# Gemini — map transcript to prompt(s) for route()
#
# If the item matches a known YOLO class → output the exact class name.
# If the item is real but unknown to YOLO → output a short descriptive phrase
#   suitable for SAM3 open-vocabulary search (e.g. "claw hammer", "hex wrench").
# ---------------------------------------------------------------------------

_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

_MAP_PROMPT = """\
You are a semantic mapper for a workshop object detector.

The user said: "{transcript}"

Map every physical item the user is looking for to either:
  (A) An exact YOLO class name from the list below — use this whenever it fits.
  (B) A short descriptive phrase for SAM3 open-vocabulary search — use this only
      when the item is NOT covered by any YOLO class.

─── YOLO CLASS NAMES (output these exactly, character-for-character) ─────────
Allen key        → hex key, L-shaped key, Allen wrench, hex wrench
Camera           → camera, cameras
Pins             → pin, pins, connector pins, header pins, jumper pins
Screwdriver Kit  → DEFAULT for any screwdriver mention ("I need a screwdriver",
                   "green case", "screwdriver box", "where do I put it back")
Screwdriver      → ONLY the bare hand tool clearly already in use
                   ("hand me that flathead", "pass me the Phillips")
ESP32            → ESP32 board, microcontroller board
Motor            → motor, DC motor, servo motor (NOT motor controller)
Drill Bits       → drill bit, drill bits, bits for a drill
Clutter          → trash, junk, cleanup intent, "what's not needed"
Brush            → brush, brushes
Slider           → slider, slide component, linear slider
Tape             → tape, tape holder, masking tape, tape roll
Motor Controllers → motor controller, ESC, speed controller,
                    red/pink plastic bag or package (controllers are shipped in them)
──────────────────────────────────────────────────────────────────────────────

DISAMBIGUATION:
- "screwdriver" → Screwdriver Kit (default); bare-tool-in-hand context → Screwdriver
- "motor" / "DC motor" → Motor; "controller" / "ESC" / "red bag" → Motor Controllers
- "drill bits" → Drill Bits; "drill" or "drilling machine" alone → NOT in list (use phrase)
- Cleanup/trash intent → Clutter

FOR ITEMS NOT IN THE YOLO LIST:
  Output a short, visually descriptive phrase that SAM3 can use to find the object
  in an image. Be specific about colour, shape, or material if helpful.
  Examples:
    "drill" / "drilling machine"  → "power drill"
    "hammer"                      → "claw hammer"
    "multimeter"                  → "multimeter"
    "soldering iron"              → "soldering iron"
    "pliers"                      → "pliers"
    "wire stripper"               → "wire stripper"

OUTPUT RULES:
- One item per line. Nothing else — no bullets, no explanations, no blank lines.
- YOLO match → exact YOLO class name (case-sensitive, spaces included).
- Non-YOLO item → short descriptive phrase (2-4 words max).
- Noise, greetings, or completely unrelated speech → output nothing.

EXAMPLES:
  "where is my Allen key"              → Allen key
  "find the cameras"                   → Camera
  "I need some pins"                   → Pins
  "pass me a screwdriver"              → Screwdriver Kit
  "hand me that flathead you're using" → Screwdriver
  "where is the ESP32"                 → ESP32
  "find the motor"                     → Motor
  "where are my drill bits"            → Drill Bits
  "where is my drill"                  → power drill
  "clean up the workspace"             → Clutter
  "find the brush"                     → Brush
  "where is the slider"                → Slider
  "I need some tape"                   → Tape
  "where are the motor controllers"    → Motor Controllers
  "find my hammer"                     → claw hammer
  "where is the multimeter"            → multimeter
  "where is the wrench and camera":
    wrench
    Camera
"""


def map_to_prompts(transcript: str) -> list:
    """
    Returns a list of prompt strings to pass to route() one at a time.
    Each string is either an exact YOLO class name or a SAM3-friendly phrase.
    """
    prompt = _MAP_PROMPT.format(transcript=transcript)
    try:
        response = _gemini_model.generate_content(prompt)
        lines = [ln.strip() for ln in response.text.strip().splitlines() if ln.strip()]
        for ln in lines:
            print(f"[gemini]  → \"{ln}\"")
        return lines
    except Exception as e:
        print(f"[gemini]  API error ({e}), skipping detection")
        return []


# ---------------------------------------------------------------------------
# Grab a frame from the Raspberry Pi TCP stream or local webcam
# ---------------------------------------------------------------------------

def grab_frame(test_image_path: str = None) -> bytes:
    if test_image_path:
        return Path(test_image_path).read_bytes()

    try:
        with socket.create_connection((PI_HOST, PI_PORT), timeout=2) as sock:
            raw_len = _recvall(sock, 4)
            msg_len = struct.unpack(">L", raw_len)[0]
            return _recvall(sock, msg_len)
    except Exception:
        pass

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("No camera available (Pi unreachable and no local webcam)")
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _recvall(sock, n: int) -> bytes:
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket closed early")
        data += packet
    return data


# ---------------------------------------------------------------------------
# Display / save result image
# ---------------------------------------------------------------------------

def show_result(composite: np.ndarray, prompts: list):
    frame_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    cv2.imwrite("result.png", frame_bgr)
    print("[saved]   → result.png")


# ---------------------------------------------------------------------------
# Entry point — accepts a transcript string directly
# ---------------------------------------------------------------------------

def run(transcript: str, image_path: str = None):
    """
    Main entry point. Call this with an already-transcribed string.
    image_path is optional; falls back to Pi stream then local webcam.
    """
    print(f"[transcript] \"{transcript}\"")

    prompts = map_to_prompts(transcript)
    if not prompts:
        print("[skip]  Nothing detectable in transcript.")
        return

    print("[camera]  Grabbing frame...")
    img_bytes = grab_frame(image_path)

    yolo = modal.Cls.from_name("detect-tools-combined", "YoloSam2Detector")()
    sam3 = modal.Cls.from_name("detect-tools-combined", "Sam3Detector")()

    # Assign a distinct color per prompt (single prompt always gets green)
    prompt_colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(prompts)}

    print(f"[modal]   Detecting {prompts} in parallel...")
    def detect(prompt):
        return prompt, route(prompt, img_bytes, yolo, sam3, color=prompt_colors[prompt])

    results = {}
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        for prompt, result in pool.map(detect, prompts):
            results[prompt] = result

    # Composite all overlays onto one image.
    # Each route() drew its masks on top of the original image — wherever pixels
    # differ from the original, those are overlays we want to keep.
    original = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    composite = original.copy()
    all_detections = []

    for prompt in prompts:
        result = results[prompt]
        result_img = np.array(Image.open(io.BytesIO(base64.b64decode(result["image"]))).convert("RGB"))
        diff_mask = np.any(result_img != original, axis=2)
        composite[diff_mask] = result_img[diff_mask]
        all_detections.extend(result.get("detections", []))

        count = result.get("count", 0)
        print(f"[result]  '{prompt}' → {count} detection(s)")
        for d in result.get("detections", []):
            print(f"            {d['label']}  {d['score']:.0%}")

    show_result(composite, prompts)

    position = {}
    if len(prompts) == 1 and results[prompts[0]].get("count") == 1:
        r = results[prompts[0]]
        position = {"cx": r["cx"], "cy": r["cy"]}
        print(f"[position] ({r['cx']}, {r['cy']})")

    # Encode as JPEG for fast transport back to the frontend
    buf = io.BytesIO()
    Image.fromarray(composite).save(buf, format="JPEG", quality=90)

    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "count": len(all_detections),
        "detections": [
            {"label": d["label"], "score": round(d["score"], 2)}
            for d in all_detections
        ],
        **position,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python voice_agent.py \"<transcript>\" [image_path]")
        sys.exit(1)
    transcript = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    run(transcript, image_path)
