import io
import base64
import socket
import struct
import numpy as np
import sounddevice as sd
import whisper
import torch
import cv2
import modal
from PIL import Image

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

def run_detection(tool: str, img_bytes: bytes) -> dict:
    model = modal.Cls.from_name("detect-tools-clip", "DetectTools")()
    return model.detect.remote([tool], img_bytes)


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
            print("[camera]    Grabbing frame...")
            img_bytes = grab_frame()

            print("[modal]     Running detection...")
            result = run_detection(tool, img_bytes)

            count = result.get("count", 0)
            print(f"[result]    {count} detection(s)")
            for d in result.get("detections", []):
                print(f"            {d['label']}  CLIP {d['clip_score']:.0%}  DINO {d['dino_score']:.0%}")

            show_result(result, tool)
            print()


if __name__ == "__main__":
    main()
