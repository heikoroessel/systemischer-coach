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
# HTML served inline - no static directory needed

DEFAULT_PROMPT = {
    "role": "Du bist ein systemischer Coach, der auf den Inhalten des Podcasts 'Systemisch Denken' von Heiko Roessel basiert. Du sprichst Deutsch und verhaeltst dich wie ein erfahrener systemischer Coach.",
    "behavior": "- Du fuehrst echte Coaching-Gespraeche, keine Beratung\n- Du stellst oeffnende Fragen statt Ratschlaege zu geben\n- Du arbeitest mit zirkulaeren Fragen, Reframing, Ressourcenorientierung\n- Du bist empathisch, praesent und wertschaetzend\n- Stelle maximal eine Frage auf einmal\n- Sprich in einem zügigen, klaren Tempo - nicht zu langsam\n- Kurze, klare Saetze",
    "interventions": "- Nach 3-4 Fragen teilst du eine Beobachtung oder ein Muster das du erkennst\n- Du spiegelst Woerter die du immer wieder hoerst\n- Bei passenden Themen empfiehlst du proaktiv eine Podcast-Episode\n- Wenn jemand nach Episoden fragt: nenne den Titel und frage welchen Link du senden soll - Website, Spotify oder Apple Podcasts\n- Verwende IMMER den gespeicherten Link aus der Datenbank - erfinde NIEMALS einen Link",
    "principles": "- Jeder Mensch hat alle Ressourcen die er braucht\n- Probleme entstehen im Kontext - Loesungen auch\n- Du arbeitest loesungs- und ressourcenorientiert\n- Du fragst nach Ausnahmen, Wundern und kleinen Schritten",
    "greeting": "Beginne herzlich. Stelle dich kurz als digitaler Coach vor, der auf dem Podcast Systemisch Denken von Heiko Roessel basiert. Frage was den Gespraechspartner heute bewegt.",
    "voice": "coral"
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
    """Load episodes from registry - compact format for prompt."""
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
                # Compact: title | W:url S:url A:url
                line = f"{num}|{title}" if num else title
                links = []
                if web:   links.append(f"W:{web}")
                if spot:  links.append(f"S:{spot}")
                if apple: links.append(f"A:{apple}")
                if links: line += "|" + "|".join(links)
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

EPISODEN (Format: Nummer|Titel|W:WebLink|S:SpotifyLink|A:AppleLink):
Verwende NUR diese Links, erfinde keine. Lies Links NIE vor - sage "Den Link findest du im Gespraechsverlauf."
Nenne Episodennummern auf Deutsch (301=dreihunderteins).
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

INDEX_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
  <title>Systemischer Coach</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --orange: #F47C20;
      --black: #1a1a1a;
      --white: #ffffff;
      --light: #f8f8f8;
      --mid: #999999;
      --border: #e8e8e8;
    }
    html, body {
      height: 100%; min-height: 100vh;
      font-family: 'Barlow', sans-serif;
      font-weight: 300;
      background: var(--white);
      color: var(--black);
      -webkit-font-smoothing: antialiased;
    }

    /* PAGE */
    .page {
      max-width: 420px;
      margin: 0 auto;
      padding: 48px 28px 60px;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }

    /* HEADER */
    .header {
      text-align: center;
      margin-bottom: 52px;
    }
    .header-title {
      font-size: 18px;
      font-weight: 400;
      color: var(--black);
      letter-spacing: .01em;
      margin-bottom: 6px;
    }
    .header-sub {
      font-size: 12px;
      font-weight: 300;
      color: var(--mid);
      letter-spacing: .04em;
      line-height: 1.6;
    }

    /* ORB */
    .orb-wrap {
      width: 120px;
      height: 120px;
      position: relative;
      cursor: pointer;
      margin-bottom: 20px;
      -webkit-tap-highlight-color: transparent;
    }
    .orb-outer {
      position: absolute; inset: 0;
      border-radius: 50%;
      border: 1px solid var(--border);
      transition: border-color .3s;
    }
    .orb-inner {
      position: absolute;
      inset: 12px;
      border-radius: 50%;
      background: var(--light);
      display: flex; align-items: center; justify-content: center;
      transition: background .3s;
    }
    .orb-icon {
      font-size: 28px;
      transition: all .3s;
    }

    /* States */
    .orb-wrap.active .orb-outer { border-color: var(--orange); }
    .orb-wrap.active .orb-inner { background: var(--orange); }
    .orb-wrap.speaking .orb-inner { animation: breathe 1.2s ease-in-out infinite; }
    .orb-wrap.listening .orb-outer { animation: ring-pulse 2s ease-in-out infinite; }
    @keyframes breathe {
      0%,100% { transform: scale(1); }
      50% { transform: scale(1.06); }
    }
    @keyframes ring-pulse {
      0%,100% { border-color: var(--orange); opacity: 1; }
      50% { border-color: var(--orange); opacity: .4; }
    }

    /* STATUS */
    .status {
      font-size: 11px;
      font-weight: 300;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--mid);
      margin-bottom: 40px;
      min-height: 16px;
      transition: color .3s;
    }
    .status.active { color: var(--orange); }

    /* MAIN BUTTON */
    .main-btn {
      width: 100%;
      padding: 15px 24px;
      font-family: 'Barlow', sans-serif;
      font-size: 13px;
      font-weight: 400;
      letter-spacing: .1em;
      text-transform: uppercase;
      border: 1px solid var(--black);
      background: transparent;
      color: var(--black);
      cursor: pointer;
      transition: all .2s;
      -webkit-tap-highlight-color: transparent;
      margin-bottom: 48px;
    }
    .main-btn.running {
      background: var(--black);
      color: var(--white);
      border-color: var(--black);
    }
    .main-btn:disabled { opacity: .3; cursor: not-allowed; }



    /* DIVIDER */
    .divider {
      width: 1px;
      height: 32px;
      background: var(--border);
      margin: 0 auto 32px;
    }

    /* TRANSCRIPT */
    .transcript { width: 100%; }
    .transcript-label {
      font-size: 10px;
      font-weight: 400;
      letter-spacing: .16em;
      text-transform: uppercase;
      color: var(--mid);
      margin-bottom: 20px;
      text-align: center;
    }
    .msg { margin-bottom: 20px; }
    .msg-who {
      font-size: 9px;
      font-weight: 400;
      letter-spacing: .2em;
      text-transform: uppercase;
      color: var(--mid);
      margin-bottom: 5px;
    }
    .msg.coach .msg-who { color: var(--orange); }
    .msg-text {
      font-size: 15px;
      font-weight: 300;
      line-height: 1.65;
      color: var(--black);
    }
    .msg-text a {
      color: var(--orange);
      text-decoration: none;
      border-bottom: 1px solid rgba(244,124,32,.3);
      word-break: break-all;
    }
    .msg-text a:active { opacity: .7; }
    .msg.user .msg-text { color: #888; }

    /* FOOTER */
    .footer {
      margin-top: 48px;
      font-size: 11px;
      font-weight: 300;
      color: var(--border);
      letter-spacing: .06em;
      text-align: center;
    }
    .footer a { color: var(--mid); text-decoration: none; }
  </style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <div class="header-title">Systemischer Coach</div>
    <div class="header-sub">
      Basierend auf dem Podcast<br>
      <em>Systemisch Denken</em> von Heiko Rössel
    </div>
  </div>

  <!-- ORB -->
  <div class="orb-wrap idle" id="orb" onclick="toggleSession()">
    <div class="orb-outer"></div>
    <div class="orb-inner">
      <span class="orb-icon" id="orbIcon">▶</span>
    </div>
  </div>

  <!-- STATUS -->
  <div class="status" id="status">Tippen zum Starten</div>

  <!-- BUTTON -->
  <button class="main-btn" id="mainBtn" onclick="toggleSession()">
    Gespräch beginnen
  </button>



  <!-- TRANSCRIPT -->
  <div id="transcriptWrap" style="display:none; width:100%">
    <div class="divider"></div>
    <div class="transcript">
      <div class="transcript-label">Gesprächsverlauf</div>
      <div id="transcript"></div>
    </div>
  </div>

  <div class="footer">
    <a href="/admin">Admin</a>
  </div>

</div>

<script>
  let pc=null, dc=null, stream=null, audio=null, active=false;
  const VOICE = 'coral';
  const SPEED = 'normal';

  const orb=document.getElementById('orb');
  const orbIcon=document.getElementById('orbIcon');
  const statusEl=document.getElementById('status');
  const mainBtn=document.getElementById('mainBtn');
  const transcript=document.getElementById('transcript');
  const transcriptWrap=document.getElementById('transcriptWrap');

  function setStatus(t, isActive=false){
    statusEl.textContent=t;
    statusEl.className='status'+(isActive?' active':'');
  }
  function setOrb(state){
    orb.className='orb-wrap '+state;
    orbIcon.textContent = state==='active'||state==='listening' ? '◼' :
                          state==='speaking' ? '◉' : '▶';
  }

  function linkify(text){
    const escaped = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return escaped.replace(/(https?:\\/\\/[^\\s\\)\\]<]+)/g,
      '<a href="$1" target="_blank" rel="noopener">$1</a>');
  }

  function addMsg(role, text){
    if(!text.trim()) return;
    const d=document.createElement('div');
    d.className='msg '+role;
    d.innerHTML=`<div class="msg-who">${role==='coach'?'Coach':'Du'}</div><div class="msg-text">${linkify(text)}</div>`;
    transcript.appendChild(d);
    d.scrollIntoView({behavior:'smooth', block:'end'});
  }

  async function toggleSession(){
    if(active){ await endSession(); }
    else { await startSession(); }
  }

  async function startSession(){
    setStatus('Verbinde …', true);
    mainBtn.disabled=true;
    try{
      const res=await fetch('/api/session',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({voice: VOICE, speed: SPEED})
      });
      if(!res.ok) throw new Error('Session-Fehler '+res.status);
      const data=await res.json();
      const key=data.client_secret?.value;
      if(!key) throw new Error('Kein Token erhalten');

      stream=await navigator.mediaDevices.getUserMedia({audio:true});
      pc=new RTCPeerConnection();
      audio=document.createElement('audio'); audio.autoplay=true;
      pc.ontrack=e=>{ audio.srcObject=e.streams[0]; };
      stream.getAudioTracks().forEach(t=>pc.addTrack(t,stream));
      dc=pc.createDataChannel('oai-events');
      dc.onmessage=handleEvent;

      const offer=await pc.createOffer();
      await pc.setLocalDescription(offer);
      const sdpRes=await fetch(
        'https://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
        { method:'POST',
          headers:{'Authorization':'Bearer '+key,'Content-Type':'application/sdp'},
          body:offer.sdp }
      );
      if(!sdpRes.ok) throw new Error('OpenAI Fehler '+sdpRes.status);
      await pc.setRemoteDescription({type:'answer', sdp:await sdpRes.text()});

      active=true;
      setOrb('listening');
      setStatus('Hört zu …', true);
      mainBtn.textContent='Beenden';
      mainBtn.className='main-btn running';
      mainBtn.disabled=false;
      transcriptWrap.style.display='block';

    }catch(err){
      console.error(err);
      setStatus('Fehler: '+err.message);
      mainBtn.disabled=false;
      setOrb('idle');
      if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
    }
  }

  async function endSession(){
    if(dc) dc.close();
    if(pc) pc.close();
    if(stream) stream.getTracks().forEach(t=>t.stop());
    if(audio) audio.srcObject=null;
    pc=dc=stream=audio=null;
    active=false;
    setOrb('idle');
    setStatus('Gespräch beendet.');
    mainBtn.textContent='Gespräch beginnen';
    mainBtn.className='main-btn';
    mainBtn.disabled=false;
  }

  function handleEvent(evt){
    let e; try{ e=JSON.parse(evt.data); }catch{ return; }
    switch(e.type){
      case 'input_audio_buffer.speech_started':
        setOrb('listening'); setStatus('Ich höre …', true); break;
      case 'input_audio_buffer.speech_stopped':
        setOrb('active'); setStatus('Denke nach …', true); break;
      case 'response.audio.started':
        setOrb('speaking'); setStatus('Coach spricht …', true); break;
      case 'response.audio_transcript.done':
        if(e.transcript) addMsg('coach', e.transcript);
        setOrb('listening'); setStatus('Hört zu …', true); break;
      case 'conversation.item.input_audio_transcription.completed':
        if(e.transcript) addMsg('user', e.transcript); break;
      case 'error':
        console.error(e.error); setStatus('Fehler aufgetreten.'); break;
    }
  }
</script>
</body>
</html>
"""

ADMIN_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0"/>
  <title>Admin – Systemischer Coach</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --orange: #F47C20;
      --black: #111;
      --white: #fff;
      --grey: #f4f4f4;
      --mid: #888;
      --green: #27ae60;
      --red: #e74c3c;
    }
    html, body { font-family: 'Barlow', sans-serif; font-weight: 300; background: var(--light); color: var(--black); -webkit-font-smoothing: antialiased; }

    header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 20px 24px; background: var(--white);
      border-bottom: 1px solid #ebebeb;
      position: sticky; top: 0; z-index: 10;
    }
    .brand-main { font-size: 15px; font-weight: 400; letter-spacing: .01em; }
    .brand-sub { font-size: 11px; font-weight: 300; color: var(--mid); letter-spacing: .04em; }
    .back { font-size: 12px; font-weight: 300; color: var(--mid); text-decoration: none; letter-spacing: .06em; }
    .back:hover { color: var(--orange); }

    /* PASSWORD GATE */
    .pw-gate {
      padding: 60px 28px; text-align: center; max-width: 380px; margin: 0 auto;
    }
    .pw-gate p { font-size: 13px; font-weight: 300; color: var(--mid); margin-bottom: 28px; }
    .pw-input {
      width: 100%; max-width: 320px; padding: 14px 16px;
      font-family: 'Barlow', sans-serif; font-size: 16px;
      border: 2px solid var(--black); background: var(--white);
      outline: none; display: block; margin: 0 auto 12px;
    }
    .pw-input:focus { border-color: var(--orange); }
    .pw-btn {
      width: 100%; max-width: 320px;
      padding: 16px; font-family: 'Barlow', sans-serif;
      font-size: 14px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase;
      background: var(--orange); color: var(--white); border: none; cursor: pointer;
    }

    /* TABS */
    .tabs {
      display: flex; background: var(--white);
      border-bottom: 1px solid #ebebeb;
      overflow-x: auto; -webkit-overflow-scrolling: touch;
    }
    .tab {
      padding: 14px 20px; font-family: 'Barlow', sans-serif;
      font-size: 11px; font-weight: 300; letter-spacing: .1em; text-transform: uppercase;
      color: var(--mid); border: none; background: transparent; cursor: pointer;
      white-space: nowrap; border-bottom: 2px solid transparent; margin-bottom: -1px;
      -webkit-tap-highlight-color: transparent;
    }
    .tab.active { color: var(--black); border-bottom-color: var(--orange); font-weight: 400; }

    .tab-content { display: none; padding: 20px; }
    .tab-content.active { display: block; }

    /* CARDS */
    .card { background: var(--white); padding: 24px; margin-bottom: 12px; border: 1px solid #ebebeb; }
    .card-title {
      font-size: 10px; font-weight: 400; letter-spacing: .18em; text-transform: uppercase;
      color: var(--mid); margin-bottom: 20px;
    }

    /* STATS */
    .stat-val { font-size: 36px; font-weight: 900; color: var(--orange); }
    .stat-label { font-size: 11px; color: var(--mid); text-transform: uppercase; letter-spacing: .1em; }

    /* FORM */
    .field { margin-bottom: 16px; }
    .field label { display: block; font-size: 11px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: var(--mid); margin-bottom: 6px; }
    input[type=text], textarea {
      width: 100%; padding: 12px 14px;
      border: 2px solid #e0e0e0; background: var(--white);
      font-family: 'Barlow', sans-serif; font-size: 15px; color: var(--black);
      outline: none; resize: vertical; -webkit-appearance: none;
    }
    input:focus, textarea:focus { border-color: var(--orange); }
    textarea { min-height: 100px; }

    /* DROPZONE */
    .dropzone {
      border: 2px dashed #ccc; padding: 28px 20px; text-align: center; cursor: pointer;
      transition: border-color .2s, background .2s;
      -webkit-tap-highlight-color: transparent;
    }
    .dropzone.drag, .dropzone:active { border-color: var(--orange); background: rgba(244,124,32,.04); }
    .dropzone-icon { font-size: 28px; margin-bottom: 8px; }
    .dropzone p { font-size: 14px; color: var(--mid); line-height: 1.5; }
    .dropzone strong { color: var(--orange); }
    .file-name { margin-top: 8px; font-size: 13px; font-weight: 700; color: var(--orange); }
    #fileInput { display: none; }

    /* STEPS */
    .steps { margin-top: 16px; display: none; }
    .progress-bar { height: 4px; background: #eee; margin-bottom: 12px; }
    .progress-fill { height: 100%; width: 0%; background: var(--orange); transition: width .4s; }
    .step { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 14px; color: var(--mid); }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #ddd; flex-shrink: 0; }
    .step.active { color: var(--black); font-weight: 700; }
    .step.active .dot { background: var(--orange); }
    .step.done { color: var(--green); font-weight: 700; }
    .step.done .dot { background: var(--green); }

    /* RESULT */
    .result { display: none; margin-top: 16px; padding: 16px; border: 2px solid; }
    .result.success { border-color: var(--green); }
    .result.error   { border-color: var(--red); }
    .result-title { font-size: 12px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 12px; }
    .result.success .result-title { color: var(--green); }
    .result.error   .result-title { color: var(--red); }
    .result-row { display: flex; justify-content: space-between; font-size: 14px; padding: 5px 0; border-bottom: 1px solid #f0f0f0; }
    .result-row:last-child { border: none; }
    .result-row b { color: var(--orange); }
    .preview-text { margin-top: 10px; font-size: 13px; color: var(--mid); line-height: 1.5; background: var(--grey); padding: 10px; }

    /* BTNS */
    .btn {
      display: block; width: 100%; padding: 16px;
      font-family: 'Barlow', sans-serif; font-size: 14px; font-weight: 900;
      letter-spacing: .08em; text-transform: uppercase;
      border: none; cursor: pointer; text-align: center;
      -webkit-tap-highlight-color: transparent;
      transition: opacity .15s;
    }
    .btn:active { opacity: .8; }
    .btn:disabled { opacity: .4; }
    .btn-orange { background: var(--orange); color: var(--white); border: none; }
    .btn-black  { background: var(--black); color: var(--white); border: none; }
    .btn-outline { background: transparent; color: var(--black); border: 1px solid #ccc; font-weight: 300; }
    .btn + .btn { margin-top: 10px; }

    /* EPISODE LIST */
    .ep-item { background: var(--white); padding: 16px; margin-bottom: 8px; border-left: 3px solid var(--orange); }
    .ep-title { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
    .ep-links { font-size: 12px; color: var(--mid); }
    .ep-edit-btn { font-size: 12px; font-weight: 700; color: var(--orange); border: none; background: none; cursor: pointer; padding: 0; margin-top: 8px; }

    /* EDIT PANEL */
    .edit-panel { display: none; background: var(--grey); padding: 16px; margin-top: 10px; }
    .edit-panel.open { display: block; }

    /* SAVE BANNER */
    .save-ok { display: none; background: var(--green); color: var(--white); padding: 12px 16px; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-top: 12px; }

    /* PROMPT SECTIONS */
    .prompt-group { margin-bottom: 20px; }
    .prompt-group h3 { font-size: 12px; font-weight: 900; letter-spacing: .15em; text-transform: uppercase; color: var(--orange); margin-bottom: 4px; }
    .hint { font-size: 12px; color: var(--mid); margin-bottom: 8px; font-style: italic; }
  </style>
</head>
<body>

<header>
  <div class="logo"></div>
  <div>
    <div class="brand-main">Admin</div>
    <div class="brand-sub">Systemischer Coach</div>
  </div>
  <a href="/" class="back">← Coach</a>
</header>

<!-- PASSWORD GATE -->
<div id="pwGate" class="pw-gate">
  <h2>Anmelden</h2>
  <p>Bitte Admin-Passwort eingeben</p>
  <input type="password" class="pw-input" id="pwInput" placeholder="Passwort" onkeydown="if(event.key==='Enter')doLogin()"/>
  <button class="pw-btn" onclick="doLogin()">Weiter</button>
  <p id="pwError" style="color:var(--red);font-size:13px;margin-top:12px;display:none">Falsches Passwort</p>
</div>

<!-- MAIN APP (hidden until login) -->
<div id="mainApp" style="display:none">
  <div class="tabs">
    <button class="tab active" onclick="showTab('upload',this)">Episoden</button>
    <button class="tab" onclick="showTab('episodes',this)">Bearbeiten</button>
    <button class="tab" onclick="showTab('prompt',this)">Coach-Prompt</button>
    <button class="tab" onclick="showTab('stats',this)">Statistiken</button>
  </div>

  <!-- UPLOAD TAB -->
  <div class="tab-content active" id="tab-upload">
    <div class="card">
      <div class="card-title">Episode hochladen</div>

      <div class="field">
        <label>Episoden-Titel</label>
        <input type="text" id="epTitle" placeholder="z.B. PSD 315 – Singularität"/>
      </div>
      <div class="field">
        <label>Nummer (optional)</label>
        <input type="text" id="epNumber" placeholder="315"/>
      </div>
      <div class="field">
        <label>Link Website</label>
        <input type="text" id="epWeb" placeholder="https://www.heikoroessel.com/..."/>
      </div>
      <div class="field">
        <label>Link Spotify</label>
        <input type="text" id="epSpotify" placeholder="https://open.spotify.com/..."/>
      </div>
      <div class="field">
        <label>Link Apple Podcasts</label>
        <input type="text" id="epApple" placeholder="https://podcasts.apple.com/..."/>
      </div>

      <div class="field">
        <label>MP3-Datei (auch über 25 MB)</label>
        <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
          <div class="dropzone-icon">🎙</div>
          <p><strong>Tippen zum Auswählen</strong><br>oder MP3 hierher ziehen<br>Große Dateien werden automatisch aufgeteilt</p>
          <div class="file-name" id="fileName"></div>
        </div>
        <input type="file" id="fileInput" accept=".mp3"/>
      </div>

      <div class="steps" id="steps">
        <div class="progress-bar"><div class="progress-fill" id="progFill"></div></div>
        <div class="step" id="s1"><div class="dot"></div>MP3 wird hochgeladen</div>
        <div class="step" id="s2"><div class="dot"></div>Whisper transkribiert</div>
        <div class="step" id="s3"><div class="dot"></div>Text in Chunks aufgeteilt</div>
        <div class="step" id="s4"><div class="dot"></div>Embeddings erstellt</div>
        <div class="step" id="s5"><div class="dot"></div>In Pinecone gespeichert</div>
      </div>

      <div class="result" id="result">
        <div class="result-title" id="resultTitle"></div>
        <div id="resultBody"></div>
      </div>

      <button class="btn btn-orange" id="uploadBtn" onclick="doUpload()" style="margin-top:16px">Jetzt verarbeiten</button>
    </div>
  </div>

  <!-- EPISODES TAB -->
  <div class="tab-content" id="tab-episodes">
    <div style="padding-bottom:8px">
      <button class="btn btn-outline" onclick="loadEpisodes()" style="margin-bottom:16px">Episoden laden</button>
      <div id="epList"><p style="font-size:14px;color:var(--mid)">Klicke "Episoden laden" um die Liste anzuzeigen.</p></div>
    </div>
  </div>

  <!-- PROMPT TAB -->
  <div class="tab-content" id="tab-prompt">
    <div class="card">
      <div class="card-title">Coach-Prompt konfigurieren</div>
      <p class="hint" style="margin-bottom:16px">Der Prompt wird dauerhaft in Pinecone gespeichert – bleibt auch nach Code-Updates erhalten.</p>

      <div class="prompt-group">
        <h3>Rolle</h3>
        <p class="hint">Wer ist dieser Coach? Auf welchem Wissen basiert er?</p>
        <textarea id="pRole" rows="4"></textarea>
      </div>
      <div class="prompt-group">
        <h3>Verhalten & Gesprächsführung</h3>
        <p class="hint">Wie verhält er sich? Welche Regeln gelten?</p>
        <textarea id="pBehavior" rows="6"></textarea>
      </div>
      <div class="prompt-group">
        <h3>Interventionen & Episoden</h3>
        <p class="hint">Wann bringt er Beobachtungen? Wie empfiehlt er Episoden?</p>
        <textarea id="pInterventions" rows="6"></textarea>
      </div>
      <div class="prompt-group">
        <h3>Systemische Prinzipien</h3>
        <textarea id="pPrinciples" rows="5"></textarea>
      </div>
      <div class="prompt-group">
        <h3>Begrüßung</h3>
        <textarea id="pGreeting" rows="3"></textarea>
      </div>

      <button class="btn btn-outline" onclick="loadPrompt()">Aktuellen Prompt laden</button>
      <button class="btn btn-black" onclick="savePrompt()" style="margin-top:10px">Prompt speichern</button>
      <div class="save-ok" id="saveOk">✓ Gespeichert – aktiv beim nächsten Gespräch</div>
    </div>
  </div>

  <!-- STATS TAB -->
  <div class="tab-content" id="tab-stats">
    <div class="card">
      <div class="card-title">Datenbank</div>
      <div style="display:flex;gap:32px;margin-bottom:20px">
        <div><div class="stat-val" id="statVec">—</div><div class="stat-label">Vektoren</div></div>
        <div><div class="stat-val" id="statEp">—</div><div class="stat-label">Episoden ca.</div></div>
      </div>
      <button class="btn btn-orange" onclick="loadStats()">Aktualisieren</button>
    </div>
  </div>
</div>

<script>
  let PW = '';

  // LOGIN
  async function doLogin(){
    const pw = document.getElementById('pwInput').value.trim();
    if(!pw) return;
    try{
      const r = await fetch('/api/stats?password='+encodeURIComponent(pw));
      if(!r.ok){ document.getElementById('pwError').style.display='block'; return; }
      PW = pw;
      document.getElementById('pwGate').style.display='none';
      document.getElementById('mainApp').style.display='block';
      loadStats();
    }catch(e){
      document.getElementById('pwError').style.display='block';
    }
  }
  document.getElementById('pwInput').addEventListener('keydown', e=>{ if(e.key==='Enter') doLogin(); });

  // TABS
  function showTab(id, btn){
    document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.getElementById('tab-'+id).classList.add('active');
    btn.classList.add('active');
  }

  // DROPZONE
  const dz=document.getElementById('dropzone');
  const fi=document.getElementById('fileInput');
  dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag');});
  dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
  dz.addEventListener('drop',e=>{
    e.preventDefault(); dz.classList.remove('drag');
    if(e.dataTransfer.files[0]){fi.files=e.dataTransfer.files; showFile(e.dataTransfer.files[0]);}
  });
  fi.addEventListener('change',()=>{ if(fi.files[0]) showFile(fi.files[0]); });
  function showFile(f){ document.getElementById('fileName').textContent=f.name+' · '+(f.size/1024/1024).toFixed(1)+' MB'; }

  // STEPS
  function setStep(n){
    for(let i=1;i<=5;i++){
      const el=document.getElementById('s'+i);
      el.className='step'+(i<n?' done':i===n?' active':'');
    }
    document.getElementById('progFill').style.width=Math.min(((n-1)/4)*95,95)+'%';
  }

  // UPLOAD
  async function doUpload(){
    const title=document.getElementById('epTitle').value.trim();
    const file=fi.files[0];
    if(!title){alert('Bitte Episoden-Titel eingeben');return;}
    if(!file){alert('Bitte MP3 auswählen');return;}

    document.getElementById('uploadBtn').disabled=true;
    document.getElementById('steps').style.display='block';
    document.getElementById('result').style.display='none';
    setStep(1);

    try{
      const fd=new FormData();
      fd.append('file',file);
      fd.append('episode_title',title);
      fd.append('episode_number',document.getElementById('epNumber').value);
      fd.append('link_website',document.getElementById('epWeb').value);
      fd.append('link_spotify',document.getElementById('epSpotify').value);
      fd.append('link_apple',document.getElementById('epApple').value);
      fd.append('password',PW);

      setStep(2);
      const res=await fetch('/api/upload-mp3',{method:'POST',body:fd});
      setStep(4);
      if(!res.ok){const e=await res.json();throw new Error(e.detail||'Fehler');}
      const d=await res.json();
      setStep(5);
      document.getElementById('progFill').style.width='100%';
      // Mark last step done
      document.getElementById('s5').className='step done';

      const r=document.getElementById('result');
      r.className='result success'; r.style.display='block';
      document.getElementById('resultTitle').textContent='Erfolgreich verarbeitet';
      document.getElementById('resultBody').innerHTML=`
        <div class="result-row"><span>Episode</span><b>${d.episode}</b></div>
        <div class="result-row"><span>Teile</span><b>${d.parts}</b></div>
        <div class="result-row"><span>Zeichen</span><b>${d.transcript_length.toLocaleString()}</b></div>
        <div class="result-row"><span>Vektoren</span><b>${d.vectors}</b></div>
        <div class="preview-text">${d.preview}</div>
      `;
      loadStats();
    }catch(err){
      const r=document.getElementById('result');
      r.className='result error'; r.style.display='block';
      document.getElementById('resultTitle').textContent='Fehler';
      document.getElementById('resultBody').innerHTML=`<div style="color:var(--red);font-size:14px">${err.message}</div>`;
    }finally{
      document.getElementById('uploadBtn').disabled=false;
    }
  }

  // STATS
  async function loadStats(){
    try{
      const r=await fetch('/api/stats?password='+encodeURIComponent(PW));
      if(!r.ok)return;
      const d=await r.json();
      document.getElementById('statVec').textContent=d.total_vectors.toLocaleString();
      document.getElementById('statEp').textContent='~'+Math.round(d.total_vectors/8);
    }catch(e){}
  }

  // EPISODES
  async function loadEpisodes(){
    const list=document.getElementById('epList');
    list.innerHTML='<p style="font-size:14px;color:var(--mid)">Lade …</p>';
    try{
      const r=await fetch('/api/episodes?password='+encodeURIComponent(PW));
      if(!r.ok)throw new Error('Fehler');
      const d=await r.json();
      if(!d.episodes.length){list.innerHTML='<p style="font-size:14px;color:var(--mid)">Noch keine Episoden gefunden.</p>';return;}
      list.innerHTML='';
      d.episodes.forEach(ep=>{
        const div=document.createElement('div');
        div.className='ep-item';
        div.innerHTML=`
          <div class="ep-title">${ep.episode} ${ep.episode_number?'#'+ep.episode_number:''}</div>
          <div class="ep-links">
            ${ep.link_website?'🌐 Website':'❌ Kein Website-Link'} &nbsp;
            ${ep.link_spotify?'🎵 Spotify':'❌ Kein Spotify'} &nbsp;
            ${ep.link_apple?'🍎 Apple':'❌ Kein Apple'}
          </div>
          <button class="ep-edit-btn" onclick="toggleEdit(this,'${ep.id}')">Links bearbeiten ▼</button>
          <div class="edit-panel" id="ep-${ep.id}">
            <div class="field"><label>Website</label><input type="text" id="ew-${ep.id}" value="${ep.link_website||''}"/></div>
            <div class="field"><label>Spotify</label><input type="text" id="es-${ep.id}" value="${ep.link_spotify||''}"/></div>
            <div class="field"><label>Apple</label><input type="text" id="ea-${ep.id}" value="${ep.link_apple||''}"/></div>
            <button class="btn btn-orange" onclick="saveEpisode('${ep.id}','${ep.episode}')" style="margin-top:8px">Speichern</button>
          </div>
        `;
        list.appendChild(div);
      });
    }catch(e){list.innerHTML='<p style="color:var(--red);font-size:14px">Fehler beim Laden</p>';}
  }

  function toggleEdit(btn, id){
    const panel=document.getElementById('ep-'+id);
    const open=panel.classList.toggle('open');
    btn.textContent=open?'Schließen ▲':'Links bearbeiten ▼';
  }

  async function saveEpisode(id, episode){
    try{
      const r=await fetch('/api/episodes/update',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          password:PW, id,episode,
          link_website:document.getElementById('ew-'+id).value,
          link_spotify:document.getElementById('es-'+id).value,
          link_apple:document.getElementById('ea-'+id).value,
        })});
      if(!r.ok)throw new Error('Fehler');
      alert('Gespeichert!');
      loadEpisodes();
    }catch(e){alert('Fehler: '+e.message);}
  }

  // PROMPT
  async function loadPrompt(){
    try{
      const r=await fetch('/api/prompt?password='+encodeURIComponent(PW));
      if(!r.ok){alert('Fehler beim Laden');return;}
      const d=await r.json();
      document.getElementById('pRole').value=d.role||'';
      document.getElementById('pBehavior').value=d.behavior||'';
      document.getElementById('pInterventions').value=d.interventions||'';
      document.getElementById('pPrinciples').value=d.principles||'';
      document.getElementById('pGreeting').value=d.greeting||'';
    }catch(e){alert('Fehler: '+e.message);}
  }

  async function savePrompt(){
    try{
      const r=await fetch('/api/prompt',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          password:PW,
          role:document.getElementById('pRole').value,
          behavior:document.getElementById('pBehavior').value,
          interventions:document.getElementById('pInterventions').value,
          principles:document.getElementById('pPrinciples').value,
          greeting:document.getElementById('pGreeting').value,
        })});
      if(!r.ok){alert('Fehler beim Speichern');return;}
      const ok=document.getElementById('saveOk');
      ok.style.display='block';
      setTimeout(()=>ok.style.display='none',4000);
    }catch(e){alert('Fehler: '+e.message);}
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def coach():
    return HTMLResponse(INDEX_HTML)

@app.get("/admin", response_class=HTMLResponse)
async def admin():
    return HTMLResponse(ADMIN_HTML)

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
    prompt = build_prompt(config, speed)
    logger.info(f"Prompt length: {len(prompt)} chars")
    request_body = {
        "model": "gpt-4o-realtime-preview",
        "voice": voice,
        "instructions": prompt,
        "input_audio_transcription": {"model": "whisper-1"},
        "turn_detection": {"type": "server_vad", "threshold": 0.6,
                           "prefix_padding_ms": 400, "silence_duration_ms": 1000}
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=request_body, timeout=30)
    if not resp.is_success:
        logger.error(f"OpenAI error {resp.status_code}: {resp.text}")
        logger.error(f"Prompt length was: {len(prompt)} chars")
        # Return error to frontend so user sees it
        return JSONResponse(status_code=200, content={"error": f"OpenAI {resp.status_code}: {resp.text}"})
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
