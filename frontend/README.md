# ToolFinder — Frontend

A makerspace vision assistant. Speak a query like *"where is my hammer"*, and the system identifies the tool's location across live camera feeds using a Modal-hosted detection backend.

---

## Stack

- **Vite** — dev server and bundler
- **Tailwind CSS** — via CDN, utility-first styling
- **Web Speech API** — browser-native speech-to-text (Chrome/Edge only)
- **WebSockets** — camera streams, pointer control, detection results
- **Vanilla JS** — no framework, modular ES modules

---

## Project Structure

```
index.html
src/
├── main.js                  # Entry point — mounts all components
├── style.css                # Fonts, keyframes, aspect-ratio (only what Tailwind can't do)
├── api/
│   ├── websocket.js         # Generic WebSocket manager (auto-reconnect, send queue)
│   ├── cameras.js           # Camera feed subscriptions (up to 4 simultaneous streams)
│   ├── pointer.js           # Pointer on/off control signal
│   └── detection.js         # Detection results listener + REST helpers
└── components/
    ├── header.js            # Title bar, connection badge, pointer toggle button
    ├── cameraGrid.js        # Live feed grid with add / remove / port-switch per slot
    ├── speech.js            # Voice query panel (Web Speech API → backend)
    └── results.js           # Detection results panel (labels, scores, annotated frame)
```

---

## Getting Started

```bash
npm create vite@latest toolfinder -- --template vanilla
cd toolfinder

# Replace the generated src/ and index.html with the files in this repo
# Then:
npm install
npm run dev
```

Open `http://localhost:5173`.

> **Note:** Speech recognition requires **Chrome or Edge**. Firefox does not support the Web Speech API.

---

## WebSocket Endpoints

The frontend expects the following WebSocket servers to be running locally. All reconnect automatically if the server is unavailable.

| Port | Purpose | Direction | Message Format |
|------|---------|-----------|---------------|
| `9999` | Camera 1 feed | Server → Client | JPEG blob |
| `10000` | Camera 2 feed | Server → Client | JPEG blob |
| `10001` | Camera 3 feed | Server → Client | JPEG blob |
| `10002` | Camera 4 feed | Server → Client | JPEG blob |
| `9998` | Pointer control | Client → Server | `{"type":"pointer","active":true\|false}` |
| `9997` | Detection results | Server → Client | JSON (see below) |

### Detection Result Schema

```json
{
  "count": 2,
  "detections": [
    { "label": "red hammer", "score": 0.92, "camera": 0 },
    { "label": "wrench",     "score": 0.87, "camera": 1 }
  ],
  "image": "<base64-encoded JPEG> | null"
}
```

If `image` is provided, the results panel will display the annotated frame.

---

## REST API

Base URL: `http://localhost:8000`

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/detect` | `{ transcript, tools[] }` | Send a voice transcript for backend processing |
| `GET` | `/status` | — | Health check (optional) |

The `sendTranscript()` function in `api/detection.js` calls `POST /detect` automatically after each recognized speech result containing a known tool name.

---

## Configuration

All configurable values are at the top of their respective files — no `.env` needed for the frontend.

| File | Variable | Default | Description |
|------|----------|---------|-------------|
| `api/cameras.js` | `CAMERA_PORTS` | `[9999, 10000, 10001, 10002]` | WebSocket ports for each camera slot |
| `api/pointer.js` | `POINTER_URL` | `ws://localhost:9998` | Pointer control socket URL |
| `api/detection.js` | `DETECTION_WS_URL` | `ws://localhost:9997` | Detection results socket URL |
| `api/detection.js` | `REST_BASE_URL` | `http://localhost:8000` | Backend REST base URL |
| `components/speech.js` | `TOOLS` | `[hammer, wrench, …]` | Tool keywords to detect in transcripts — **keep in sync with Python `TOOLS` list** |

---

## UI Overview

### Header
- **TOOLFINDER** title with live pulse indicator
- **Status badge** — updates automatically: `READY` / `CONNECTED` / `LISTENING` / `DETECTING…` / `DISCONNECTED`
- **POINTER ON/OFF** button — toggles the pointer state and broadcasts it to the backend over `:9998`

### Camera Grid
- Starts with 1 feed open on port `9999`
- **+ ADD FEED** opens additional slots (up to 4)
- Each slot has a **port switcher** dropdown to change which camera it displays
- Each slot has a **✕** remove button to close that feed
- Frame counter and live clock shown per slot

### Voice Query
- **START LISTENING** activates the microphone via Web Speech API
- Interim results appear in real time as you speak
- On a final result, known tool names are extracted and highlighted as tags
- The transcript and detected tools are automatically sent to `POST /detect`

### Detections
- Updates in real time as results arrive over `ws://localhost:9997`
- Shows each detected tool with its confidence score and which camera it was found on
- If the backend returns an annotated image, it is displayed below the results list

---

## Adding a New Tool

1. Add the tool name to `TOOLS` in `src/components/speech.js`
2. Add the same name to the Python `TOOLS` list in your backend script
3. That's it — the speech extractor and result renderer handle everything else dynamically

---

## Backend

This frontend pairs with a Python backend that runs:

- **Silero VAD + Whisper** — transcription (handled by the browser here, but the backend can also accept raw audio)
- **Gemini** — natural language parsing of the transcript to extract tool descriptors
- **Modal** — serverless GPU inference for object detection / SAM segmentation
- **Raspberry Pi** — camera stream source over TCP, rebroadcast as WebSocket blobs