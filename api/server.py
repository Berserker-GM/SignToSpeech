"""FastAPI backend for Sign Language to Speech web app."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.inference import BROWSER_VOICE, SignEngine, correct_sentence, get_available_voices, speak_text

engine: SignEngine | None = None
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    engine = SignEngine()
    yield
    engine = None


app = FastAPI(title="Sign Language to Speech API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CorrectRequest(BaseModel):
    words: list[str]


class SpeakRequest(BaseModel):
    text: str
    voice_id: str = BROWSER_VOICE["id"]


@app.get("/api/health")
def health():
    return {"ok": True, "models_loaded": engine is not None}


@app.get("/api/voices")
def voices():
    return {"voices": get_available_voices(), "default_voice_id": BROWSER_VOICE["id"]}


@app.post("/api/correct")
def api_correct(body: CorrectRequest):
    if not body.words:
        raise HTTPException(400, "No words provided")
    return correct_sentence(body.words)


@app.post("/api/speak")
def api_speak(body: SpeakRequest):
    if not body.text.strip():
        raise HTTPException(400, "Empty text")
    try:
        return speak_text(body.text.strip(), body.voice_id)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/session/reset")
def reset_session():
    if engine:
        engine.reset_session()
    return {"ok": True}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    if not engine:
        await websocket.close(code=1011)
        return

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "frame")

            if msg_type == "set_mode":
                engine.mode = data.get("mode", "AUTO")
                await websocket.send_json({"type": "mode", "mode": engine.mode})
                continue

            if msg_type == "clear":
                engine.sentence.clear()
                await websocket.send_json({"type": "cleared", "sentence": []})
                continue

            if msg_type == "reset":
                engine.reset_session()
                await websocket.send_json({"type": "reset"})
                continue

            landmarks = data.get("landmarks")
            if not landmarks or len(landmarks) != 63:
                await websocket.send_json(
                    {"type": "frame", "hand_detected": False, "sentence": engine.sentence}
                )
                continue

            now = time.time()
            result = engine.process_frame(landmarks, now)
            await websocket.send_json({"type": "frame", "hand_detected": True, **result})
    except WebSocketDisconnect:
        pass


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api") or full_path.startswith("ws"):
            raise HTTPException(404)
        index = FRONTEND_DIST / "index.html"
        if full_path and (FRONTEND_DIST / full_path).is_file():
            return FileResponse(FRONTEND_DIST / full_path)
        return FileResponse(index)


def run():
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
