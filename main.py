"""
Systemischer Coach - FastAPI Backend v2
Features: MP3-Splitter, Prompt-Editor, Episoden-Links, Stimme/Geschwindigkeit
"""
import os, re, uuid, json, math, logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai, tiktoken, httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Systemischer Coach API v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX", "systemischer-coach")
ADMIN_PASSWORD   = os.environ.get("ADMIN_PASSWORD", "coach2024")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
PROMPT_FILE = Path("coach_prompt.json")

app.mount("/static", StaticFiles(directory="static"), name="static")

# helpers
def get_index():
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(name=PINECONE_INDEX, dimension=1536, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    return pc.Index(PINECONE_INDEX)

def chunk_text(text, episode_title, chunk_size=500, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks, start, idx = [], 0, 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append({"id": f"{uuid.uuid4().hex}_{idx}",
                       "text": enc.decode(tokens[start:end]),
                       "episode": episode_title, "chunk_index": idx})
        start += chunk_size - overlap
        idx += 1
    return chunks

def embed(texts):
    r = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [i.embedding for i in r.data]

def transcribe(path):
    with open(path, "rb") as f:
        return openai_client.audio.transcriptions.create(
            model="whisper-1", file=f, language="de", response_format="text")

def split_mp3(path, max_bytes=24*1024*1024):
    data = path.read_bytes()
    if len(data) <= max_bytes:
        return [path]
    n = math.ceil(len(data) / max_bytes)
    sz = len(data) // n
    parts = []
    for i in range(n):
        p = path.parent / f"{path.stem}_p{i}.mp3"
        p.write_bytes(data[i*sz : (i+1)*sz if i < n-1 else len(data)])
        parts.append(p)
    return parts

def load_prompt():
    default = {
        "role": "Du bist ein systemischer Coach, der auf den Inhalten des Podcasts 'Systemisch Denken' von Heiko Roessel basiert. Du sprichst Deutsch und verhaeltst dich wie ein erfahrener systemischer Coach.",
        "behavior": "- Du fuehrst echte Coaching-Gespraeche, keine Beratung\n- Du stellst oeffnende Fragen statt Ratschlaege zu geben\n- Du arbeitest mit zirkulaeren Fragen, Reframing, Ressourcenorientierung\n- Du bist empathisch, praesent und wertschaetzend\n- Stelle maximal eine Frage auf einmal\n- Sprich in kurzen Saetzen fuer den Sprachdialog",
        "interventions": "- Nach 3-4 Fragen teilst du eine Beobachtung oder ein Muster das du erkennst\n- Du spiegelst Woerter die du immer wieder hoerst\n- Bei passenden Themen empfiehlst du proaktiv eine Podcast-Episode\n- Wenn jemand nach Episoden fragt, empfiehlst du gezielt mit Titel und fragst welchen Link du senden sollst: Website, Spotify oder Apple Podcasts",
        "principles": "- Jeder Mensch hat alle Ressourcen die er braucht\n- Probleme entstehen im Kontext - Loesungen auch\n- Du arbeitest loesungs- und ressourcenorientiert\n- Du fragst nach Ausnahmen, Wundern und kleinen Schritten",
        "greeting": "Beginne mit einer herzlichen Begruassung. Stelle dich kurz als digitaler Coach vor, der auf dem Podcast Systemisch Denken von Heiko Roessel basiert. Frage dann was den Gesprachspartner heute bewegt.",
        "voice": "alloy",
        "speed": 1.2
    }
    if PROMPT_FILE.exists():
        try:
            saved = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
            default.update(saved)
        except Exception:
            pass
    return default

def save_prompt(config):
    PROMPT_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

def build_prompt(config=None):
    if config is None:
        config = load_prompt()
    return f"""
{config.get('role','')}

VERHALTEN:
{config.get('behavior','')}

INTERVENTIONEN UND EPISODEN:
{config.get('interventions','')}

SYSTEMISCHE PRINZIPIEN:
{config.get('principles','')}

{config.get('greeting','')}
""".strip()

# routes
@app.get("/", response_class=HTMLResponse)
async def coach():
    return HTMLResponse(open("static/index.html", encoding="utf-8").read())

@app.get("/admin", response_class=HTMLResponse)
async def admin():
    return HTMLResponse(open("static/admin.html", encoding="utf-8").read())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/session")
async def session(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    config = load_prompt()
    voice = body.get("voice", config.get("voice", "alloy"))
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-realtime-preview-2024-12-17",
                "voice": voice,
                "instructions": build_prompt(config),
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad", "threshold": 0.5,
                                   "prefix_padding_ms": 300, "silence_duration_ms": 700}
            }, timeout=30)
    return JSONResponse(content=resp.json())

@app.get("/api/prompt")
async def get_prompt(password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    return load_prompt()

@app.post("/api/prompt")
async def set_prompt(request: Request):
    data = await request.json()
    if data.get("password") != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    config = {k: v for k, v in data.items() if k != "password"}
    save_prompt(config)
    return {"success": True}

@app.post("/api/upload-mp3")
async def upload(
    file: UploadFile = File(...),
    episode_title: str = Form(...),
    episode_number: str = Form(""),
    link_website: str = Form(""),
    link_spotify: str = Form(""),
    link_apple: str = Form(""),
    password: str = Form(...),
):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Nur MP3 erlaubt")

    tmp = Path(f"/tmp/{uuid.uuid4().hex}.mp3")
    parts = []
    try:
        tmp.write_bytes(await file.read())
        parts = split_mp3(tmp)
        extra = [p for p in parts if p != tmp]
        logger.info(f"Split into {len(parts)} part(s)")

        transcript = " ".join(str(transcribe(p)) for p in parts).strip()
        logger.info(f"Transcribed: {len(transcript)} chars")

        tmp.unlink(missing_ok=True)
        for p in extra:
            p.unlink(missing_ok=True)

        chunks = chunk_text(transcript, episode_title)
        index = get_index()
        total = 0
        for i in range(0, len(chunks), 50):
            batch = chunks[i:i+50]
            vecs = [{"id": c["id"], "values": emb,
                     "metadata": {"text": c["text"], "episode": episode_title,
                                  "episode_number": episode_number,
                                  "link_website": link_website,
                                  "link_spotify": link_spotify,
                                  "link_apple": link_apple,
                                  "chunk_index": c["chunk_index"]}}
                    for c, emb in zip(batch, embed([c["text"] for c in batch]))]
            index.upsert(vectors=vecs)
            total += len(vecs)

        return {"success": True, "episode": episode_title, "parts": len(parts),
                "transcript_length": len(transcript), "chunks": len(chunks),
                "vectors": total,
                "preview": transcript[:500] + ("..." if len(transcript) > 500 else "")}

    except Exception as e:
        tmp.unlink(missing_ok=True)
        for p in parts:
            try: p.unlink(missing_ok=True)
            except: pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def stats(password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    try:
        s = get_index().describe_index_stats()
        return {"total_vectors": s.total_vector_count, "index_name": PINECONE_INDEX}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
