import io
import base64
import os
import sys
import socket
import struct
from pathlib import Path
import numpy as np
import sounddevice as sd
import whisper
import torch
import cv2
import modal
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your setup
# ---------------------------------------------------------------------------

# Exact class names that the YOLO model was trained on.
# Gemini maps all conversational speech to one of these.
YOLO_CLASSES = [
    "Allen key", "Sensor Case", "Camera", "Pins", "Screwdriver Kit",
    "ESP32", "Screwdriver", "Motor", "Drill Bits", "Clutter",
    "Brush", "Slider", "Tape", "Motor Controllers",
]

SAMPLE_RATE      = 16000   # Whisper expects 16kHz
CHUNK_SAMPLES    = 512     # silero-vad minimum required chunk size at 16kHz

SPEECH_THRESHOLD  = 0.65   # raised: reduces false triggers from background noise
SILENCE_CHUNKS    = 20     # consecutive silent chunks before utterance ends (~600ms)
MIN_SPEECH_CHUNKS = 25     # raised: require ~0.8s of speech before transcribing
MAX_SPEECH_SEC    = 8      # hard cap — cuts runaway recordings

PI_HOST = "192.168.50.232"
PI_PORT = 9999

# Image path: pass as first CLI arg (e.g. python voice_agent.py test.jpg)
# Set to None to use live Pi/webcam feed instead
TEST_IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else None

WHISPER_MODEL = "base"     # tiny / base / small — trade speed for accuracy

DISPLAY_MS = 5000          # how long to show result image (ms)


# ---------------------------------------------------------------------------
# 1. Load models (called once at startup)
# ---------------------------------------------------------------------------

def load_models():
    print("[startup] Loading silero-VAD...")
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )
    vad_model.eval()

    print(f"[startup] Loading Whisper {WHISPER_MODEL}...")
    whisper_model = whisper.load_model(WHISPER_MODEL)

    print("[startup] Models ready. Listening...\n")
    return vad_model, whisper_model


# ---------------------------------------------------------------------------
# 2. Listen — blocks until a complete utterance is captured
# ---------------------------------------------------------------------------

def listen_for_speech(vad_model, stream) -> np.ndarray:
    """
    Runs silero-VAD on mic chunks from an already-open stream.
    Returns a float32 numpy array at 16kHz once the worker
    stops speaking (SILENCE_CHUNKS of consecutive silence).
    """
    speech_buffer = []
    speech_chunks  = 0
    silence_chunks = 0
    in_speech      = False

    while True:
        chunk, _ = stream.read(CHUNK_SAMPLES)
        chunk_1d  = chunk[:, 0]                  # (512,) float32

        chunk_tensor = torch.from_numpy(chunk_1d).unsqueeze(0)  # (1, 512)
        with torch.no_grad():
            confidence = vad_model(chunk_tensor, SAMPLE_RATE).item()

        if confidence >= SPEECH_THRESHOLD:
            in_speech      = True
            silence_chunks = 0
            speech_chunks += 1
            speech_buffer.append(chunk_1d)

            # Hard cap
            if speech_chunks >= MAX_SPEECH_SEC * SAMPLE_RATE / CHUNK_SAMPLES:
                break

        elif in_speech:
            silence_chunks += 1
            speech_buffer.append(chunk_1d)   # include trailing silence for Whisper

            if silence_chunks >= SILENCE_CHUNKS:
                break                         # utterance ended

    if speech_chunks < MIN_SPEECH_CHUNKS:
        return None                               # too short — noise, not speech

    return np.concatenate(speech_buffer)          # flat float32 array


# ---------------------------------------------------------------------------
# 3. Transcribe with Whisper
# ---------------------------------------------------------------------------

def transcribe(whisper_model, audio_buf: np.ndarray) -> str:
    result = whisper_model.transcribe(
        audio_buf,
        fp16=False,
        language="en",                          # stops random-language hallucinations
        initial_prompt="where is the Allen key, find the sensor case, where is the camera, find my pins, screwdriver kit, ESP32, motor, drill bits, brush, tape, motor controllers, screwdriver",
    )
    return result["text"].strip().lower()


# ---------------------------------------------------------------------------
# 4. Gemini — map conversational speech to exact YOLO class names
#    Gemini is the ONLY stage that decides what to detect. It understands
#    context (e.g. "screwdriver" vs "screwdriver kit"), synonyms, and
#    natural phrasing. It outputs only class names the YOLO model knows.
# ---------------------------------------------------------------------------

_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

_MAP_PROMPT = """\
You are a semantic mapper for a YOLO workshop object detector.

The user spoke this sentence (converted from speech to text): "{transcript}"

Your task: identify every physical item the user is looking for, then map each one
to the single matching YOLO class name from the list below. Output one class name
per line — nothing else.

─── VALID YOLO CLASS NAMES ───────────────────────────────────────────────────
Allen key       → hex key, L-shaped key, Allen wrench, hex wrench, "the key"
Sensor Case     → case/box for sensors, sensor storage, "put sensors back"
Camera          → camera, cameras
Pins            → pin, pins, connector pins, header pins, jumper pins
Screwdriver Kit → DEFAULT for any screwdriver mention: "I need a screwdriver",
                  "screwdriver box", "green case", "where do I put screwdrivers back",
                  "find me a screwdriver" (user will get it from the kit)
Screwdriver     → ONLY when context clearly means the bare hand tool in use:
                  "pass me that screwdriver", "hand me the flathead",
                  "the Phillips head I am already using"
ESP32           → ESP32 board, esp32 device, microcontroller board
Motor           → motor, DC motor, servo motor (NOT motor controller)
Drill Bits      → drill bit, drill bits, bits for a drill
                  NOTE: "drill" or "drilling machine" alone is NOT in vocabulary
Clutter         → trash, junk, things to clean up, items to remove,
                  "what's not needed", "clean my workspace", "locate the trash"
Brush           → brush, brushes
Slider          → slider, slide component
Tape            → tape, tape holder, red tape, masking tape, tape roll
Motor Controllers → motor controller, ESC, speed controller,
                    pink or red plastic bag/wrapper/package/pouch
                    (motor controllers are shipped inside these)
──────────────────────────────────────────────────────────────────────────────

DISAMBIGUATION RULES (read carefully):
1. Screwdriver vs Screwdriver Kit:
   - DEFAULT: any mention of "screwdriver" maps to Screwdriver Kit — because in a
     workshop, when someone says "I need a screwdriver" they go to the kit to get one.
   - EXCEPTION: use Screwdriver ONLY if context clearly shows they mean the bare tool
     already in hand or in immediate use (e.g. "hand me the flathead", "pass me that
     screwdriver you're holding").
2. Motor vs Motor Controllers:
   - "motor", "DC motor", "spinning part" → Motor
   - "motor controller", "ESC", "pink/red bag/package/wrapper" → Motor Controllers
3. Drill Bits vs drill:
   - "drill bits", "bits", "bits for drilling" → Drill Bits
   - "drill", "drilling machine" alone → NOT in YOLO vocabulary → output: Drill
4. Clutter: the user will never say "clutter" directly. Map any cleanup/removal
   intent (trash, unnecessary items, clean workspace) to Clutter.
5. If the user mentions a REAL workshop tool (wrench, pliers, hammer, soldering
   iron, oscilloscope, multimeter, etc.) that is NOT in the class list above,
   output the exact tool name extracted from their speech with the first letter
   capitalized (e.g. "Hammer", "Wrench", "Pen", "Drill").
6. If the text is random noise, greetings, or completely unrelated, output nothing.

OUTPUT RULES:
- One item per line.
- If the tool matches a YOLO class: output the exact YOLO class name (case-sensitive).
- If the tool is real but NOT in the YOLO class list: output the tool name with
  first letter capitalized (e.g. Hammer, Wrench, Pliers, Drill).
- No explanations, no extra text, no blank lines, no bullet points.
- Multiple items = multiple lines.

EXAMPLES:
  "where is my Allen key"                    → Allen key
  "I can't find the key"                     → Allen key
  "where is the case for my sensors"         → Sensor Case
  "find my cameras"                          → Camera
  "I need some pins"                         → Pins
  "where is the green case"                  → Screwdriver Kit
  "pass me a screwdriver"                    → Screwdriver Kit
  "hand me the flathead you're holding"      → Screwdriver
  "where is the ESP32"                       → ESP32
  "find the motor"                           → Motor
  "where are my drill bits"                  → Drill Bits
  "where is my drill"                        → Drill
  "I need to clean up, what's not needed"    → Clutter
  "find the brush"                           → Brush
  "where is the slider"                      → Slider
  "I need some tape"                         → Tape
  "where are the motor controllers"          → Motor Controllers
  "where is my red package"                  → Motor Controllers
  "find my hammer"                           → Hammer
  "where is my wrench"                       → Wrench
  "where is the wrench and camera":
    Wrench
    Camera
  "find my pen"                              → Pen
"""


def map_to_yolo_classes(transcript: str) -> list:
    """
    Sends the transcript to Gemini. Returns a list of items extracted from
    the user's speech — either exact YOLO class names or capitalized tool
    names for items outside the YOLO vocabulary.
    Returns an empty list if the speech contains nothing detectable.
    Falls back to an empty list on API error.
    """
    prompt = _MAP_PROMPT.format(transcript=transcript)
    try:
        response = _gemini_model.generate_content(prompt)
        lines = [ln.strip() for ln in response.text.strip().splitlines() if ln.strip()]
        for ln in lines:
            print(f"[gemini]    → \"{ln}\"")
        return lines
    except Exception as e:
        print(f"[gemini]    API error ({e}), skipping detection")
        return []


# ---------------------------------------------------------------------------
# 5. Grab a single frame from the Raspberry Pi TCP stream
#    Falls back to local webcam if the Pi is not reachable.
# ---------------------------------------------------------------------------

def grab_frame() -> bytes:
    # Test mode: use a local image file instead of any camera
    if TEST_IMAGE_PATH:
        with open(TEST_IMAGE_PATH, "rb") as f:
            return f.read()

    # Try Pi TCP stream first (same length-prefix protocol as compRasiCode/camStream.py)
    try:
        with socket.create_connection((PI_HOST, PI_PORT), timeout=2) as sock:
            raw_len = _recvall(sock, 4)
            msg_len = struct.unpack(">L", raw_len)[0]
            img_bytes = _recvall(sock, msg_len)
        return img_bytes
    except Exception:
        pass

    # Fallback: local webcam
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
# 6. Run Modal detection
# ---------------------------------------------------------------------------

def run_detection(yolo_classes: list, img_bytes: bytes) -> dict:
    # YOLO class names are already the right queries — pass them directly.
    model = modal.Cls.from_name("detect-tools-sam3", "DetectTools")()
    return model.detect.remote(yolo_classes, img_bytes, yolo_classes)


# ---------------------------------------------------------------------------
# 7. Display result image
# ---------------------------------------------------------------------------

def show_result(result: dict, tool: str):
    if result.get("image") is None:
        print(f"  No detection returned for '{tool}'")
        return

    img_bytes = base64.b64decode(result["image"])
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    img_stem   = Path(TEST_IMAGE_PATH).stem if TEST_IMAGE_PATH else "frame"
    tool_slug  = tool.replace(" ", "_").replace(",", "")
    out_name   = f"result_{img_stem}_{tool_slug}.png"
    cv2.imwrite(out_name, frame)
    print(f"  Saved → {out_name}")
    cv2.imshow(f"Found: {tool}", frame)
    cv2.waitKey(DISPLAY_MS)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 8. Main loop
# ---------------------------------------------------------------------------

def main():
    vad_model, whisper_model = load_models()

    # Open mic stream once — keeps the device claimed and avoids
    # Windows MME PortAudioError when reopening after Ctrl+C
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=CHUNK_SAMPLES) as stream:
        while True:
            print("[listening] Say something like 'where is my hammer'...")

            audio_buf = listen_for_speech(vad_model, stream)
            if audio_buf is None:
                continue   # noise / too short, keep listening

            text = transcribe(whisper_model, audio_buf)
            print(f"[heard]     \"{text}\"")

            detected = map_to_yolo_classes(text)

            if not detected:
                print("[skip]      Nothing detectable found in transcript.\n")
                continue

            print(f"[yolo]      {', '.join(detected)}")

            print("[camera]    Grabbing frame...")
            img_bytes = grab_frame()

            print("[modal]     Running detection...")
            result = run_detection(detected, img_bytes)

            count = result.get("count", 0)
            print(f"[result]    {count} detection(s)")
            for d in result.get("detections", []):
                print(f"            {d['label']}  {d['score']:.0%}")

            show_result(result, ', '.join(detected))
            print()


if __name__ == "__main__":
    main()
