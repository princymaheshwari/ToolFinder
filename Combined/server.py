import io
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from voice_agent import run

app = FastAPI()


@app.post("/detect")
async def detect(
    transcript: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Receives a transcript string and an image file from the frontend.
    Runs the full detection pipeline and returns the annotated image as PNG.
    """
    img_data = await image.read()

    # Write image to a temp file so voice_agent can read it by path
    suffix = os.path.splitext(image.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(img_data)
        tmp_path = tmp.name

    try:
        # run() is blocking (Modal calls) — offload to thread pool so the
        # event loop stays free for other requests
        result_png = await run_in_threadpool(run, transcript, tmp_path)
    finally:
        os.unlink(tmp_path)

    if result_png is None:
        raise HTTPException(status_code=422, detail="Nothing detectable in transcript.")

    return Response(content=result_png, media_type="image/png")
