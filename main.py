"""
Systemischer Coach - FastAPI Backend v3
- Prompt in Pinecone gespeichert (bleibt bei Updates erhalten)
- Episode-Metadaten nachtraeglich bearbeitbar
- MP3-Splitter fuer grosse Dateien
- Episoden-Links (Website, Spotify, Apple)
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

app = FastAPI(title="Systemischer Coach v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX", "systemischer-coach")
ADMIN_PASSWORD   = os.environ.get("ADMIN_PASSWORD", "coach2024")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")

DEFAULT_PROMPT = {
    "role": "Du bist ein systemischer Coach, der auf den Inhalten des Podcasts 'Systemisch Denken' von Heiko Roessel basiert. Du sprichst Deutsch und verhaeltst dich wie ein erfahrener systemischer Coach.",
    "behavior": "- Du fuehrst echte Coaching-Gespraeche, keine Beratung\n- Du stellst oeffnende Fragen statt Ratschlaege zu geben\n- Du arbeitest mit zirkulaeren Fragen, Reframing, Ressourcenorientierung\n- Du bist empathisch, praesent und wertschaetzend\n- Stelle maximal eine Frage auf einmal\n- Sprich in einem zügigen, klaren Tempo - nicht zu langsam\n- Kurze, klare Saetze",
    "interventions": "- Nach 3-4 Fragen teilst du eine Beobachtung oder ein Muster das du erkennst\n- Du spiegelst Woerter die du immer wieder hoerst\n- Bei passenden Themen empfiehlst du proaktiv eine Podcast-Episode\n- Wenn jemand nach Episoden fragt: nenne den Titel und frage welchen Link du senden soll - Website, Spotify oder Apple Podcasts\n- Verwende IMMER den gespeicherten Link aus der Datenbank - erfinde NIEMALS einen Link",
    "principles": "- Jeder Mensch hat alle Ressourcen die er braucht\n- Probleme entstehen im Kontext - Loesungen auch\n- Du arbeitest loesungs- und ressourcenorientiert\n- Du fragst nach Ausnahmen, Wundern und kleinen Schritten",
    "greeting": "Beginne herzlich. Stelle dich kurz als digitaler Coach vor, der auf dem Podcast Systemisch Denken von Heiko Roessel basiert. Frage was den Gespraechspartner heute bewegt.",
    "voice": "alloy"
}

# ── Pinecone ──────────────────────────────────────────────────────────────────

def get_index():
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(name=PINECONE_INDEX, dimension=1536, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    return pc.Index(PINECONE_INDEX)

def save_prompt_to_pinecone(config: dict):
    """Store prompt config as a special vector in Pinecone."""
    try:
        index = get_index()
        # Use a fixed dummy vector for config storage
        dummy = [0.0] * 1536
        dummy[0] = 1.0  # distinguish from real vectors
        index.upsert(vectors=[{
            "id": "__coach_prompt_config__",
            "values": dummy,
            "metadata": {"type": "config", "data": json.dumps(config, ensure_ascii=False)}
        }])
        logger.info("Prompt saved to Pinecone")
    except Exception as e:
        logger.error(f"Failed to save prompt: {e}")

def load_prompt_from_pinecone() -> dict:
    """Load prompt config from Pinecone."""
    try:
        index = get_index()
        result = index.fetch(ids=["__coach_prompt_config__"])
        if result and result.vectors and "__coach_prompt_config__" in result.vectors:
            data = result.vectors["__coach_prompt_config__"].metadata.get("data", "{}")
            config = json.loads(data)
            merged = DEFAULT_PROMPT.copy()
            merged.update(config)
            return merged
    except Exception as e:
        logger.error(f"Failed to load prompt: {e}")
    return DEFAULT_PROMPT.copy()

def load_episode_list() -> str:
    """Load all episodes from registry and return as formatted string for prompt."""
    try:
        index = get_index()
        dummy = [0.0] * 1536
        dummy[1] = 1.0
        results = index.query(vector=dummy, top_k=300, include_metadata=True,
                              filter={"type": {"$eq": "episode_registry"}})
        lines = []
        for m in results.matches:
            md = m.metadata
            if md.get("type") == "episode_registry":
                title = md.get("episode", "")
                num = md.get("episode_number", "")
                web = md.get("link_website", "")
                spot = md.get("link_spotify", "")
                apple = md.get("link_apple", "")
                line = f"- [{num}] {title}" if num else f"- {title}"
                links = []
                if web:   links.append(f"Website: {web}")
                if spot:  links.append(f"Spotify: {spot}")
                if apple: links.append(f"Apple: {apple}")
                if links: line += " | " + " | ".join(links)
                lines.append(line)
        return "\n".join(lines) if lines else "(Noch keine Episoden)"
    except Exception as e:
        logger.error(f"Failed to load episode list: {e}")
    return "(Episoden nicht ladbar)"


def build_prompt(config=None, speed="normal") -> str:
    if config is None:
        config = load_prompt_from_pinecone()
    speed_instruction = {
        "slow":   "Sprich in einem ruhigen, langsamen Tempo mit deutlichen Pausen.",
        "normal": "Sprich in einem natuerlichen, klaren Tempo.",
        "fast":   "Sprich in einem zuegigen, energischen Tempo - keine langen Pausen."
    }.get(speed, "Sprich in einem natuerlichen Tempo.")
    episode_list = load_episode_list()
    return f"""
{config.get('role','')}

TEMPO: {speed_instruction}

VERHALTEN:
{config.get('behavior','')}

INTERVENTIONEN & EPISODEN:
{config.get('interventions','')}

EPISODEN-LINKS - STRENGE REGELN:
Du hast Zugriff auf eine exakte Liste aller Episoden mit ihren echten Links.
Verwende AUSSCHLIESSLICH diese Links - erfinde NIEMALS einen Link.
Nenne IMMER den exakten Titel aus dieser Liste.
Wenn kein Link vorhanden ist, sage das ehrlich.
WICHTIG: Lies Links NIEMALS laut vor. Sage stattdessen: "Den Link findest du direkt im Gespraechsverlauf."
Episodennummern immer auf Deutsch aussprechen: 301 = "dreihundertein", 315 = "dreihundertfuenfzehn".

VERFUEGBARE EPISODEN:
{episode_list}

SYSTEMISCHE PRINZIPIEN:
{config.get('principles','')}

{config.get('greeting','')}
""".strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        p.write_bytes(data[i*sz:(i+1)*sz if i < n-1 else len(data)])
        parts.append(p)
    return parts

# ── Routes ────────────────────────────────────────────────────────────────────

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
    config = load_prompt_from_pinecone()
    voice = body.get("voice", config.get("voice", "alloy"))
    speed = body.get("speed", "normal")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-realtime-preview-2024-12-17",
                "voice": voice,
                "instructions": build_prompt(config, speed),
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad", "threshold": 0.5,
                                   "prefix_padding_ms": 300, "silence_duration_ms": 700}
            }, timeout=30)
    return JSONResponse(content=resp.json())

@app.get("/api/prompt")
async def get_prompt(password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    return load_prompt_from_pinecone()

@app.post("/api/prompt")
async def set_prompt(request: Request):
    data = await request.json()
    if data.get("password") != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    config = {k: v for k, v in data.items() if k != "password"}
    save_prompt_to_pinecone(config)
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
    extra = []
    try:
        tmp.write_bytes(await file.read())
        parts = split_mp3(tmp)
        extra = [p for p in parts if p != tmp]
        transcript = " ".join(str(transcribe(p)) for p in parts).strip()
        tmp.unlink(missing_ok=True)
        for p in extra: p.unlink(missing_ok=True)

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

        # Save episode registry entry
        _save_episode_registry(index, episode_title, episode_number, link_website, link_spotify, link_apple)

        return {"success": True, "episode": episode_title, "parts": len(parts),
                "transcript_length": len(transcript), "chunks": len(chunks), "vectors": total,
                "preview": transcript[:500] + ("..." if len(transcript) > 500 else "")}
    except Exception as e:
        tmp.unlink(missing_ok=True)
        for p in extra:
            try: p.unlink(missing_ok=True)
            except: pass
        raise HTTPException(status_code=500, detail=str(e))

def _save_episode_registry(index, title, number, link_web, link_spotify, link_apple):
    """Save episode metadata in a registry entry for editing later."""
    try:
        ep_id = f"__episode_registry__{re.sub(r'[^a-zA-Z0-9]', '_', title)[:50]}"
        dummy = [0.0] * 1536
        dummy[1] = 1.0
        index.upsert(vectors=[{
            "id": ep_id,
            "values": dummy,
            "metadata": {
                "type": "episode_registry",
                "episode": title,
                "episode_number": number,
                "link_website": link_web,
                "link_spotify": link_spotify,
                "link_apple": link_apple,
            }
        }])
    except Exception as e:
        logger.error(f"Registry save failed: {e}")

@app.get("/api/episodes")
async def list_episodes(password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    try:
        index = get_index()
        # Query with registry dummy vector
        dummy = [0.0] * 1536
        dummy[1] = 1.0
        results = index.query(vector=dummy, top_k=200, include_metadata=True,
                              filter={"type": {"$eq": "episode_registry"}})
        episodes = []
        for m in results.matches:
            md = m.metadata
            if md.get("type") == "episode_registry":
                episodes.append({
                    "id": m.id,
                    "episode": md.get("episode", ""),
                    "episode_number": md.get("episode_number", ""),
                    "link_website": md.get("link_website", ""),
                    "link_spotify": md.get("link_spotify", ""),
                    "link_apple": md.get("link_apple", ""),
                })
        return {"episodes": episodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/episodes/update")
async def update_episode(request: Request):
    data = await request.json()
    if data.get("password") != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    try:
        index = get_index()
        ep_id = data["id"]
        index.update(id=ep_id, set_metadata={
            "link_website": data.get("link_website", ""),
            "link_spotify": data.get("link_spotify", ""),
            "link_apple":   data.get("link_apple", ""),
            "episode_number": data.get("episode_number", ""),
        })
        # Also update all content chunks for this episode
        episode_title = data.get("episode", "")
        if episode_title:
            dummy_q = [0.0] * 1536
            dummy_q[2] = 1.0
            chunks = index.query(vector=dummy_q, top_k=500, include_metadata=True,
                                 filter={"episode": {"$eq": episode_title}})
            for match in chunks.matches:
                if match.metadata.get("type") != "episode_registry":
                    index.update(id=match.id, set_metadata={
                        "link_website": data.get("link_website", ""),
                        "link_spotify": data.get("link_spotify", ""),
                        "link_apple":   data.get("link_apple", ""),
                    })
        return {"success": True}
    except Exception as e:
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
