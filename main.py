"""
Systemischer Coach - FastAPI Backend
=====================================
Handles:
- MP3 Upload & Transcription (Whisper)
- Vector Storage (Pinecone)
- OpenAI Realtime API Session Token
- Semantic Search for RAG
"""

import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
from pinecone import Pinecone, ServerlessSpec
import tiktoken

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Systemischer Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "systemischer-coach")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "coach2024")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_index():
    """Get existing Pinecone index or create a new one."""
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info(f"Created Pinecone index: {PINECONE_INDEX}")
    return pc.Index(PINECONE_INDEX)

# ── Static Files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Helper: Text Chunking ─────────────────────────────────────────────────────
def chunk_text(text: str, episode_title: str, chunk_size: int = 500, overlap: int = 50):
    """Split transcript into overlapping chunks for better retrieval."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append({
            "id": f"{episode_title}_{chunk_idx}_{uuid.uuid4().hex[:8]}",
            "text": chunk_text,
            "episode": episode_title,
            "chunk_index": chunk_idx,
        })
        start += chunk_size - overlap
        chunk_idx += 1

    return chunks

# ── Helper: Embed Text ────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create embeddings using OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

# ── Helper: Semantic Search ───────────────────────────────────────────────────
def search_knowledge(query: str, top_k: int = 5) -> list[dict]:
    """Search Pinecone for relevant podcast content."""
    try:
        index = get_or_create_index()
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

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def coach_page():
    """Serve the main coach interface."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin interface."""
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/session")
async def create_realtime_session():
    """
    Create an OpenAI Realtime API session token.
    The frontend uses this ephemeral token to connect directly to OpenAI Realtime.
    """
    try:
        # Get dynamic system prompt with knowledge base context
        system_prompt = build_system_prompt()

        response = openai_client.post(
            "/realtime/sessions",
            body={
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
            }
        )
        return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_endpoint(query: str = Form(...)):
    """Search knowledge base and return relevant chunks."""
    results = search_knowledge(query)
    return {"results": results}

@app.post("/api/upload-mp3")
async def upload_mp3(
    file: UploadFile = File(...),
    episode_title: str = Form(...),
    password: str = Form(...),
):
    """
    Admin endpoint: Upload MP3, transcribe with Whisper, store in Pinecone.
    """
    # Auth check
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")

    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Nur MP3-Dateien erlaubt")

    try:
        # 1. Save temp file
        tmp_path = Path(f"/tmp/{uuid.uuid4().hex}.mp3")
        content = await file.read()
        tmp_path.write_bytes(content)
        logger.info(f"Saved MP3: {tmp_path}, size: {len(content)} bytes")

        # 2. Transcribe with Whisper
        logger.info("Starting Whisper transcription...")
        with open(tmp_path, "rb") as audio_file:
            transcript_response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="de",
                response_format="text"
            )
        transcript_text = transcript_response
        logger.info(f"Transcription complete: {len(transcript_text)} characters")

        # 3. Clean up temp file
        tmp_path.unlink()

        # 4. Chunk the transcript
        chunks = chunk_text(transcript_text, episode_title)
        logger.info(f"Created {len(chunks)} chunks")

        # 5. Embed chunks in batches
        index = get_or_create_index()
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
            logger.info(f"Upserted batch {i//batch_size + 1}: {total_upserted} vectors total")

        return {
            "success": True,
            "episode": episode_title,
            "transcript_length": len(transcript_text),
            "chunks_created": len(chunks),
            "vectors_stored": total_upserted,
            "transcript_preview": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text,
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        # Clean up temp file if it exists
        if 'tmp_path' in locals() and tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats(password: str):
    """Get knowledge base statistics."""
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    try:
        index = get_or_create_index()
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "index_name": PINECONE_INDEX,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/episode/{episode_title}")
async def delete_episode(episode_title: str, password: str):
    """Delete all vectors for a specific episode."""
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    try:
        index = get_or_create_index()
        # Delete by metadata filter
        index.delete(filter={"episode": {"$eq": episode_title}})
        return {"success": True, "message": f"Episode '{episode_title}' gelöscht"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def build_system_prompt() -> str:
    """Build the coach system prompt."""
    return """Du bist ein systemischer Coach, der auf den Inhalten eines deutschsprachigen Coaching-Podcasts basiert. 
Du sprichst Deutsch und verhältst dich wie ein echter, erfahrener systemischer Coach.

DEINE ROLLE:
- Du führst echte Coaching-Gespräche, keine Beratungsgespräche
- Du stellst kluge, öffnende Fragen statt Ratschläge zu geben
- Du arbeitest mit systemischen Methoden: zirkuläre Fragen, Reframing, Ressourcenorientierung
- Du bist empathisch, präsent und wertschätzend
- Du hältst Pausen aus und gibst dem Gesprächspartner Raum

SYSTEMISCHE PRINZIPIEN:
- Jeder Mensch hat alle Ressourcen, die er braucht
- Probleme entstehen im Kontext – Lösungen auch
- Du arbeitest lösungs- und ressourcenorientiert
- Du fragst nach Ausnahmen, Wundern und kleinen Schritten
- Du respektierst die Autonomie des Gesprächspartners

GESPRÄCHSFÜHRUNG:
- Beginne mit einer offenen, einladenden Frage
- Höre aktiv zu und spiegele das Gehörte
- Stelle maximal eine Frage auf einmal
- Fasse gelegentlich zusammen, was du gehört hast
- Arbeite mit Metaphern und Bildern wenn passend

SPRACHE:
- Sprich natürlich und warm, nicht therapeutisch steif
- Nutze einfache, klare Sprache
- Vermeide Fachbegriffe ohne Erklärung
- Sprich in kurzen Sätzen für den Sprachdialog

Beginne das Gespräch mit einer herzlichen Begrüßung und einer einladenden ersten Frage, was den Gesprächspartner heute bewegt."""
