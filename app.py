"""Local read-aloud book reader: serves PDF pages, sentence highlights and Kokoro speech."""
import hashlib
import io
import json
import socket
import asyncio
import threading
import warnings
from collections import OrderedDict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pymupdf
import segno
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from kokoro import KModel, KPipeline
from starlette.concurrency import run_in_threadpool

from extract import extract

ROOT = Path(__file__).parent
BOOKS = ROOT / "books"
CACHE = ROOT / "cache"
PROGRESS_FILE = ROOT / "progress.json"
MY_VOICES = ROOT / "voices"  # drop a ~10 second .wav/.mp3 of a voice here to read in that voice
PORT = 8765
SAMPLE_RATE = 24000

KOKORO_VOICES = {
    "af_heart": "Heart (US)",
    "af_bella": "Bella (US)",
    "af_nicole": "Nicole (US, soft)",
    "am_michael": "Michael (US)",
    "am_fenrir": "Fenrir (US)",
    "bf_emma": "Emma (UK)",
    "bm_george": "George (UK)",
    "bm_fable": "Fable (UK)",
}
DEFAULT_VOICE = "cb:default"

BOOKS.mkdir(exist_ok=True)
MY_VOICES.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

# ---------- speech ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading voice models on {device}...")
kokoro = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
pipelines = {lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=kokoro) for lang in "ab"}

chatterbox = None
chatterbox_conds = {}
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    chatterbox = ChatterboxTurboTTS.from_pretrained(device=device)
    chatterbox_conds["default"] = chatterbox.conds
except Exception as e:  # keep working with Kokoro alone
    print(f"Chatterbox unavailable, using Kokoro only: {e}")

gpu_lock = threading.Lock()
audio_cache: OrderedDict[tuple, bytes] = OrderedDict()


def my_voice_files():
    return {p.stem: p for p in sorted(MY_VOICES.iterdir()) if p.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a")}


def all_voices():
    natural = {}
    if chatterbox:
        natural["cb:default"] = "Narrator"
        natural.update({f"cb:{name}": name for name in my_voice_files()})
    return {"natural": natural, "fast": KOKORO_VOICES}


def resolve_voice(voice: str) -> str:
    v = all_voices()
    if voice in v["natural"] or voice in v["fast"]:
        return voice
    return DEFAULT_VOICE if chatterbox else "af_heart"


def _chatterbox(text: str, voice: str) -> np.ndarray:
    name = voice[3:]
    if name not in chatterbox_conds:
        chatterbox.prepare_conditionals(str(my_voice_files()[name]))
        chatterbox_conds[name] = chatterbox.conds
    chatterbox.conds = chatterbox_conds[name]
    wav = chatterbox.generate(text)
    return wav.squeeze().cpu().numpy().astype(np.float32)


def _kokoro(text: str, voice: str) -> np.ndarray:
    pipe = pipelines["b" if voice.startswith("b") else "a"]
    chunks = [np.asarray(a.numpy() if hasattr(a, "numpy") else a)
              for _, _, a in pipe(text, voice=voice, speed=1.0, split_pattern=None) if a is not None]
    return np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE // 4, dtype=np.float32)


def synthesize(text: str, voice: str) -> bytes:
    with gpu_lock:
        pcm = _chatterbox(text, voice) if voice.startswith("cb:") else _kokoro(text, voice)
    pcm = np.concatenate([pcm, np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)])  # short gap between sentences
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


synthesize("Ready.", "af_heart")  # warm up the GPU
if chatterbox:
    synthesize("Ready.", DEFAULT_VOICE)

# ---------- books ----------
books_lock = threading.Lock()
book_data: dict[str, dict] = {}
docs: dict[str, pymupdf.Document] = {}


def book_id(path: Path) -> str:
    return hashlib.sha1(path.name.encode()).hexdigest()[:12]


def list_books():
    return {book_id(p): p for p in sorted(BOOKS.glob("*.pdf"))}


def get_path(bid: str) -> Path:
    path = list_books().get(bid)
    if not path:
        raise HTTPException(404, "Book not found")
    return path


def get_book(bid: str) -> dict:
    with books_lock:
        if bid in book_data:
            return book_data[bid]
        path = get_path(bid)
        st = path.stat()
        cache_file = CACHE / f"{bid}-{int(st.st_mtime)}-{st.st_size}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            print(f"Preparing {path.name} (first time only)...")
            data = extract(str(path))
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        data["title"] = path.stem
        book_data[bid] = data
        return data


def get_doc(bid: str) -> pymupdf.Document:
    with books_lock:
        if bid not in docs:
            docs[bid] = pymupdf.open(get_path(bid))
        return docs[bid]


def load_progress() -> dict:
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


progress_lock = threading.Lock()

# ---------- web ----------
app = FastAPI()
tts_queue = asyncio.Lock()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/books")
def books():
    progress = load_progress()
    out = []
    for bid, p in list_books().items():
        out.append({"id": bid, "title": p.stem, "progress": progress.get(bid, {})})
    return out


@app.post("/api/books")
async def upload(file: UploadFile):
    name = Path(file.filename or "book.pdf").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    (BOOKS / name).write_bytes(await file.read())
    return {"id": book_id(BOOKS / name)}


@app.get("/api/book/{bid}")
def book(bid: str):
    data = get_book(bid)
    return JSONResponse(data, headers={"Cache-Control": "no-cache"})


@app.get("/api/book/{bid}/page/{n}")
def page_image(bid: str, n: int, w: int = 1000):
    doc = get_doc(bid)
    if not 0 <= n < doc.page_count:
        raise HTTPException(404)
    w = max(300, min(w, 2000))
    with books_lock:
        page = doc[n]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(w / page.rect.width, w / page.rect.width))
        img = pix.tobytes("jpeg", jpg_quality=82)
    return Response(img, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/book/{bid}/tts/{sid}")
async def tts(bid: str, sid: int, request: Request, voice: str = DEFAULT_VOICE):
    voice = resolve_voice(voice)
    sentences = get_book(bid)["sentences"]
    if not 0 <= sid < len(sentences):
        raise HTTPException(404)
    key = (bid, sid, voice)
    wav = audio_cache.get(key)
    if wav is None:
        async with tts_queue:
            # the reader cancels clips it no longer needs (after a jump); skip those
            if await request.is_disconnected():
                return Response(status_code=499)
            wav = audio_cache.get(key) or await run_in_threadpool(synthesize, sentences[sid]["t"], voice)
        audio_cache[key] = wav
        while len(audio_cache) > 400:
            audio_cache.popitem(last=False)
    return Response(wav, media_type="audio/wav", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/sample")
async def sample(voice: str = DEFAULT_VOICE):
    text = "Here's how I sound reading your book. Tap any sentence on the page to start from there."
    wav = await run_in_threadpool(synthesize, text, resolve_voice(voice))
    return Response(wav, media_type="audio/wav")


@app.get("/api/progress/{bid}")
def get_progress(bid: str):
    return load_progress().get(bid, {})


@app.put("/api/progress/{bid}")
async def put_progress(bid: str, request: Request):
    body = await request.json()
    with progress_lock:
        progress = load_progress()
        progress[bid] = {k: body[k] for k in ("sentence", "voice", "speed", "percent") if k in body}
        PROGRESS_FILE.write_text(json.dumps(progress, indent=1), encoding="utf-8")
    return {"ok": True}


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


@app.get("/api/info")
def info():
    url = f"http://{lan_ip()}:{PORT}"
    qr = segno.make(url).svg_inline(omitsize=True, border=1, dark="#000", light="#fff")  # viewBox only, so CSS can size it
    return {"voices": all_voices(), "defaultVoice": resolve_voice(DEFAULT_VOICE),
            "voicesFolder": str(MY_VOICES), "phoneUrl": url, "qr": qr}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

if __name__ == "__main__":
    print()
    print(f"  Book reader is running.")
    print(f"  On this PC:     http://localhost:{PORT}")
    print(f"  On your phone:  http://{lan_ip()}:{PORT}   (same Wi-Fi)")
    print(f"  Close this window to stop it.")
    print()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
