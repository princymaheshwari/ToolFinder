import io
import base64
import os
import socket
import struct
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

TOOLS = [
    "hammer", "wrench", "pliers", "screwdriver", "drill",
    "saw", "chisel", "clamp", "tape measure", "level",
]

SAMPLE_RATE      = 16000   # Whisper expects 16kHz
CHUNK_SAMPLES    = 512     # silero-vad minimum required chunk size at 16kHz

SPEECH_THRESHOLD  = 0.65   # raised: reduces false triggers from background noise
SILENCE_CHUNKS    = 20     # consecutive silent chunks before utterance ends (~600ms)
MIN_SPEECH_CHUNKS = 25     # raised: require ~0.8s of speech before transcribing
MAX_SPEECH_SEC    = 8      # hard cap — cuts runaway recordings

PI_HOST = "192.168.50.232"
PI_PORT = 9999

# Set to a local image path to test without Pi/webcam (e.g. "test.jpg")
# Set to None to use live camera
TEST_IMAGE_PATH = "test.jpg"

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
        initial_prompt="where is my hammer, where is my wrench, where is my drill, find the screwdriver, locate the pliers",
    )
    return result["text"].strip().lower()


# ---------------------------------------------------------------------------
# 4. Extract tool name from transcript
# ---------------------------------------------------------------------------

def extract_tool(text: str):
    """
    Returns the first TOOLS entry found in the transcript, or None.
    e.g. "where is my wrench" → "wrench"
    """
    for tool in TOOLS:
        if tool in text:
            return tool
    return None


# ---------------------------------------------------------------------------
# 5. Gemini — get short visual description of the tool for richer DINO/CLIP
#    Results are cached so Gemini is only called once per unique tool name.
# ---------------------------------------------------------------------------

_description_cache: dict = {}
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Hardcoded fallbacks used when Gemini API is unavailable.
# Each description is deliberately shape/color-specific so DINO can distinguish
# the target tool from other objects that share its material (metal, plastic, etc).
_TOOL_FALLBACKS = {
    "hammer":       "claw hammer with flat head",
    "wrench":       "adjustable open jaw wrench",
    "pliers":       "hinged pivot jaw pliers",
    "screwdriver":  "long thin shaft screwdriver",
    "drill":        "trigger grip power drill chuck",
    "saw":          "long blade with jagged teeth",
    "chisel":       "flat beveled blade chisel",
    "clamp":        "c-shaped screw frame clamp",
    "tape measure": "yellow compact retractable tape",
    "level":        "long flat bar with bubble vial",
}

_GEMINI_SYSTEM_PROMPT = """\
You generate short visual descriptions for an AI object detector (DINO + CLIP).
The detector will search for the described object inside a photo of a cluttered workshop.

YOUR GOAL: produce a 2-4 word phrase that uniquely identifies the tool so the detector
picks the RIGHT object and ignores everything else on the workbench.

STRICT RULES — follow all of them:
1. NEVER use the words: metal, metallic, steel, iron, tool, workshop, hand, common, typical,
   large, small, object, item, device. These words apply to almost everything in a workshop
   and will cause the detector to highlight the wrong objects.
2. Focus ONLY on features that make this tool visually DISTINCT from other workshop objects:
   - Unique SHAPE or silhouette (e.g. claw on a hammer, pivot hinge on pliers, chuck on drill)
   - Distinctive COLOR if the tool has one (yellow tape, orange handle, red grip)
   - A unique structural part no other tool shares (trigger on a drill, bubble vial on a level)
3. If the user's sentence mentions a color or size modifier (e.g. "red hammer", "small pliers",
   "blue screwdriver"), you MUST include that color/size in the output — it is the most
   important distinguishing feature.
4. If no color or modifier is mentioned, do NOT invent one unless the tool almost always
   appears in a specific color (e.g. tape measures are almost always yellow/orange).
5. Output 2-4 words ONLY. No punctuation. No explanation. No extra words.

SHAPE REFERENCE (use these unique features, not generic ones):
- hammer      → claw at back + flat striking face  (NOT "metal hammer")
- pliers      → pivot hinge + two spreading handles + gripping jaw  (NOT "serrated metal jaws")
- wrench      → adjustable C-shaped jaw opening  (NOT "metal wrench")
- screwdriver → long thin narrow shaft + bulky handle  (NOT "metal screwdriver")
- drill       → cylindrical body + front chuck + trigger  (NOT "power tool")
- saw         → long flat blade + jagged teeth along one edge  (NOT "serrated tool")
- chisel      → short flat angled blade + thick striking handle
- clamp       → C or F shaped frame + threaded screw mechanism
- tape measure→ compact square case + yellow retractable ribbon
- level       → long flat rectangular bar + small bubble vial window

EXAMPLES:
  tool="hammer",       user said="where is my hammer"           → claw hammer flat head
  tool="hammer",       user said="where is my red hammer"       → red claw hammer
  tool="pliers",       user said="find the pliers"              → pivot hinge jaw pliers
  tool="pliers",       user said="where are my short pliers"    → short pivot pliers
  tool="drill",        user said="where is the drill"           → cylindrical drill with chuck
  tool="tape measure", user said="find the tape"                → yellow compact tape dispenser
  tool="screwdriver",  user said="blue screwdriver please"      → blue thin shaft screwdriver
  tool="level",        user said="where is my level"            → flat bar bubble vial level
"""

def get_tool_description(tool: str, transcript: str = "") -> str:
    """
    Calls Gemini once per (tool, transcript) and caches by tool name.
    Passes the full transcript so Gemini can extract color/size modifiers.
    Falls back to hardcoded shape-specific descriptions if the API call fails.
    """
    # If transcript has new color/size context, bypass cache to get a fresh description.
    # Otherwise use the cached value so we don't burn API quota on repeated calls.
    color_words = {"red","blue","green","yellow","orange","black","white","gray","grey",
                   "big","small","large","short","long","tiny","thick","thin"}
    has_modifier = any(w in transcript.split() for w in color_words)

    if not has_modifier and tool in _description_cache:
        return _description_cache[tool]

    user_context = f'The user said: "{transcript}"' if transcript else ""
    prompt = (
        f"{_GEMINI_SYSTEM_PROMPT}\n\n"
        f"Now generate a description for:\n"
        f"  tool = \"{tool}\"\n"
        f"  {user_context}\n"
        f"Output only the 2-4 word description:"
    )

    try:
        response = _gemini_model.generate_content(prompt)
        description = response.text.strip().lower()
        print(f"[gemini]    {tool} → \"{description}\"")
    except Exception as e:
        description = _TOOL_FALLBACKS.get(tool, tool)
        print(f"[gemini]    API error, using fallback → \"{description}\"")

    _description_cache[tool] = description
    return description


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

def run_detection(tool: str, img_bytes: bytes, description: str = None) -> dict:
    model = modal.Cls.from_name("detect-tools-clip", "DetectTools")()
    return model.detect.remote([tool], img_bytes, description)


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

    cv2.imwrite("result.png", frame)
    print(f"  Saved → result.png")
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

            tool = extract_tool(text)
            if tool is None:
                print("[skip]      No known tool found in transcript.\n")
                continue

            print(f"[tool]      {tool}")
            description = get_tool_description(tool, transcript=text)

            print("[camera]    Grabbing frame...")
            img_bytes = grab_frame()

            print("[modal]     Running detection...")
            result = run_detection(tool, img_bytes, description)

            count = result.get("count", 0)
            print(f"[result]    {count} detection(s)")
            for d in result.get("detections", []):
                print(f"            {d['label']}  CLIP {d['clip_score']:.0%}  DINO {d['dino_score']:.0%}")

            show_result(result, tool)
            print()


if __name__ == "__main__":
    main()
