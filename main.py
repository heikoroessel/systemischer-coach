"""
Systemischer Coach - FastAPI Backend
"""

import os
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Systemischer Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX  = os.environ.get("PINECONE_INDEX", "systemischer-coach")
ADMIN_PASSWORD  = os.environ.get("ADMIN_PASSWORD", "coach2024")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

app.mount("/static", StaticFiles(directory="static"), name="static")


def get_pinecone_index():
    """Lazy-load Pinecone to avoid startup errors."""
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(PINECONE_INDEX)


def chunk_text(text: str, episode_title: str, chunk_size: int = 500, overlap: int = 50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        chunks.append({
            "id": f"{episode_title}_{chunk_idx}_{uuid.uuid4().hex[:8]}",
            "text": chunk_text_str,
            "episode": episode_title,
            "chunk_index": chunk_idx,
        })
        start += chunk_size - overlap
        chunk_idx += 1
    return chunks


def embed_texts(texts: list) -> list:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def search_knowledge(query: str, top_k: int = 5) -> list:
    try:
        index = get_pinecone_index()
        query_embedding = embed_texts([query])[0]
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [
            {
                "text": match.metadata.get("text", ""),
                "episode": match.metadata.get("episode", ""),
                "score": match.score,
            }
            for match in results.matches
            if match.score > 0.3
        ]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


@app.get("/", response_class=HTMLResponse)
async def coach_page():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/session")
async def create_realtime_session():
    try:
        system_prompt = build_system_prompt()
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-realtime-preview-2024-12-17",
                    "voice": "alloy",
                    "instructions": system_prompt,
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                    }
                },
                timeout=30,
            )
        return JSONResponse(content=resp.json())
    except Exception as e:
        logger.error(f"Session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-mp3")
async def upload_mp3(
    file: UploadFile = File(...),
    episode_title: str = Form(...),
    password: str = Form(...),
):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Nur MP3-Dateien erlaubt")

    tmp_path = Path(f"/tmp/{uuid.uuid4().hex}.mp3")
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        with open(tmp_path, "rb") as audio_file:
            transcript_response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="de",
                response_format="text"
            )
        transcript_text = transcript_response
        tmp_path.unlink()

        chunks = chunk_text(transcript_text, episode_title)
        index = get_pinecone_index()
        batch_size = 50
        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embed_texts(texts)
            vectors = [
                {
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": {
                        "text": chunk["text"],
                        "episode": chunk["episode"],
                        "chunk_index": chunk["chunk_index"],
                    }
                }
                for chunk, embedding in zip(batch, embeddings)
            ]
            index.upsert(vectors=vectors)
            total_upserted += len(vectors)

        return {
            "success": True,
            "episode": episode_title,
            "transcript_length": len(transcript_text),
            "chunks_created": len(chunks),
            "vectors_stored": total_upserted,
            "transcript_preview": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text,
        }
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats(password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "index_name": PINECONE_INDEX,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def build_system_prompt() -> str:
    return """Du bist ein systemischer Coach, der auf den Inhalten eines deutschsprachigen Coaching-Podcasts basiert.
Du sprichst Deutsch und verhältst dich wie ein echter, erfahrener systemischer Coach.

DEINE ROLLE:
- Du führst echte Coaching-Gespräche, keine Beratungsgespräche
- Du stellst kluge, öffnende Fragen statt Ratschläge zu geben
- Du arbeitest mit systemischen Methoden: zirkuläre Fragen, Reframing, Ressourcenorientierung
- Du bist empathisch, präsent und wertschätzend

SYSTEMISCHE PRINZIPIEN:
- Jeder Mensch hat alle Ressourcen, die er braucht
- Probleme entstehen im Kontext – Lösungen auch
- Du arbeitest lösungs- und ressourcenorientiert
- Du fragst nach Ausnahmen, Wundern und kleinen Schritten

GESPRÄCHSFÜHRUNG:
- Stelle maximal eine Frage auf einmal
- Höre aktiv zu und spiegele das Gehörte
- Sprich in kurzen Sätzen für den Sprachdialog
- Sei warm, natürlich und menschlich

Beginne mit einer herzlichen Begrüßung und frage was den Gesprächspartner heute bewegt."""
