from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List
import uvicorn

from voice_agent import run

app = FastAPI()

# Allow requests from any origin (frontend is on a different machine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_credentials=False,  # must be False when using ""
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# WebSocket manager — pushes detection results to all connected frontends
# Mirrors results on ws://<host>:8000/ws/detection
# (Frontend detection.js should point here instead of ws://localhost:9997)
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        print(f"[ws] Client connected ({len(self._connections)} total)")

    def disconnect(self, ws: WebSocket):
        self._connections.remove(ws)
        print(f"[ws] Client disconnected ({len(self._connections)} remaining)")

    async def broadcast(self, data: dict):
        for ws in list(self._connections):
            try:
                await ws.send_json(data)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/detection")
async def ws_detection(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep-alive ping handling
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# REST endpoint — called by the frontend when a voice query is submitted
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    transcript: str
    tools: list = []    # frontend hint, unused — Gemini handles mapping


@app.post("/detect")
async def detect(req: DetectRequest, request: Request):
    """
    Receives a transcript from the frontend.
    Backend grabs a camera frame from the client's spinCam.py WebSocket stream.
    Returns annotated image as base64 JPEG + detection metadata.
    Also pushes the same result to all connected WebSocket clients.
    """
    client_ip = request.client.host
    print(f"[api]  POST /detect  transcript='{req.transcript}'  client={client_ip}")

    # run() is blocking — offload to thread pool so the event loop stays free
    result = await run_in_threadpool(run, req.transcript, None, client_ip)

    if result is None:
        raise HTTPException(status_code=422, detail="Nothing detectable in transcript.")

    # Push to any WebSocket listeners (e.g. results panel)
    await manager.broadcast(result)

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
