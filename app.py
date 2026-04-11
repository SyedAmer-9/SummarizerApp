import os
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

# ── Bootstrap ──────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please check your .env file.")

client = genai.Client(api_key=api_key)

app = FastAPI(title="StandupAI")
templates = Jinja2Templates(directory="templates")

# ── Database ───────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

KNOWN_PROJECTS = ["onblick", "reccopilot", "sales", "rig"]


def get_db(project: str) -> sqlite3.Connection:
    db_path = DATA_DIR / f"{project.lower()}.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS standups (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT NOT NULL,
            raw_notes    TEXT NOT NULL,
            summary      TEXT,
            format_type  TEXT,
            action_items TEXT,
            blockers     TEXT,
            members      TEXT,
            project      TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


# ── Word-limit enforcer ────────────────────────────────────────────────────
def enforce_word_limit(text: str, limit: int) -> str:
    """
    Hard-truncate generated text to `limit` words,
    ending at the last complete sentence within that limit.
    Only applied to paragraphs mode where the user set a word count.
    """
    words = text.split()
    if len(words) <= limit:
        return text

    # Take the first `limit` words, then walk back to a sentence boundary
    truncated = " ".join(words[:limit])
    # Find the last sentence-ending punctuation
    match = re.search(r'(.*[.!?])\s', truncated + " ", re.DOTALL)
    if match:
        return match.group(1).strip()
    return truncated.strip()


# ── Pydantic models ────────────────────────────────────────────────────────
class SummaryRequest(BaseModel):
    text: str
    length: int = 50
    format_type: str = "paragraphs"
    slide_count: Optional[int] = 3
    points_count: Optional[int] = 5
    project: str  # UPDATED: Now mandatory

# ── Prompts ────────────────────────────────────────────────────────────────

# Step 0: Strip Teams transcript noise before anything else
def parse_and_clean_transcript(text: str) -> tuple[str, list[str]]:
    """
    Instantly parses a Teams transcript using Python.
    Returns a tuple: (cleaned_text, list_of_first_names)
    """
    lines = text.split('\n')
    cleaned_lines = []
    names = set()
    
    # Noise phrases we want to instantly delete
    noise_filters = ["joined the meeting", "left the meeting", "raised hand", "can you hear me"]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 1. Strip obvious Teams system noise
        if any(noise in line.lower() for noise in noise_filters):
            continue
            
        # 2. Extract speaker names using Regex
        # Matches patterns like "Alice Kumar:" or "[10:05 AM] Alice:"
        match = re.match(r'^(?:\[.*?\]\s*)?([A-Za-z\s]+?)\s*:', line)
        if match:
            full_name = match.group(1).strip()
            # Safety check: ensure it's not a weird system message
            if len(full_name) < 25 and full_name.lower() not in ["note", "meeting", "everyone"]:
                first_name = full_name.split()[0] # Grab just the first name
                names.add(first_name)
        
        # 3. Keep the valid work lines
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines), list(names)


# Step 2: Summarise with strict word / format instructions
def build_summary_prompt(req: SummaryRequest, names: list[str], cleaned_transcript: str) -> str:
    name_section = ""
    if names:
        name_section = (
            f"These people participated: {', '.join(names)}.\n"
            "Structure the summary by grouping updates per person.\n\n"
        )

    if req.format_type == "points":
        fmt = (
            f"Format: exactly {req.points_count} bullet points using Markdown (- item).\n"
            f"Hard limit: {req.points_count} bullets. No more, no fewer.\n"
            "Each bullet should be one concise sentence."
        )
    elif req.format_type == "slides":
        fmt = (
            f"Format: exactly {req.slide_count} slides separated by --- (Markdown HR).\n"
            f"Hard limit: exactly {req.slide_count} slides. No more, no fewer.\n"
            "Each slide needs a ## heading and at least 50 words of content."
        )
    else:
        # Paragraphs — be very explicit about the word limit
        fmt = (
            f"Format: clear paragraphs in Markdown.\n"
            f"HARD WORD LIMIT: {req.length} words MAXIMUM. This is a strict ceiling, not a target.\n"
            f"Before writing, plan your response to stay under {req.length} words.\n"
            "Do not exceed this under any circumstance. Write tight, dense prose."
        )

    return (
        "You are an expert technical scribe for a software company. "
        "Summarize the following standup transcript.\n\n"
        f"{name_section}"
        f"{fmt}\n\n"
        "Transcript:\n"
        f"{cleaned_transcript}"
    )


# Step 3: Action items + blockers
ACTION_PROMPT = """\
Extract structured information from this standup transcript.

Return ONLY valid JSON — no markdown fences, no explanation, nothing else:
{{"action_items": ["...", "..."], "blockers": ["...", "..."]}}

Rules:
- action_items: concrete tasks someone explicitly committed to doing
- blockers: things actively slowing the team down right now
- Keep each item to one sentence
- Attribute items to people where possible, e.g. "Alice to review PR #42"
- If nothing found for a category, return an empty array

Transcript:
{transcript}"""


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health_check():
    try:
        client.models.count_tokens(model="gemini-2.5-flash", contents="ping")
        return {"status": "ok"}
    except Exception as e:
        print(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Gemini API unreachable")


@app.post("/summarize")
async def summarize_text(req: SummaryRequest):
    """
    SSE stream. Events in order:
      data: {"stage": "cleaning"}                        — preprocessing started
      data: {"stage": "names", "names": [...]}           — names extracted
      data: {"token": "..."}                             — summary tokens streaming
      event: action_items  data: {action_items, blockers}
      event: meta          data: {project, names, record_id}
      event: done          data: {}
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Notes cannot be empty.")

    project = req.project

    async def stream():

        
        # ── Stages 0 & 1: Instant Python Parsing ──────────────────────────
        yield f"data: {json.dumps({'stage': 'cleaning'})}\n\n"
        
        cleaned, names = parse_and_clean_transcript(req.text)
        
        yield f"data: {json.dumps({'stage': 'names', 'names': names})}\n\n"
        
        # ── Stage 2: Stream the summary (Stays exactly the same) ──────────
        summary_prompt = build_summary_prompt(req, names, cleaned)
        

        # ── Stage 2: Stream the summary ───────────────────────────────────
        summary_prompt = build_summary_prompt(req, names, cleaned)
        full_summary_parts = []

        try:
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=summary_prompt,
            ):
                token = chunk.text or ""
                full_summary_parts.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            error_msg = str(e)
            # If we hit the 429 Rate Limit
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                user_msg = "Google API rate limit reached. Please wait 10 seconds and try again."
            else:
                user_msg = f"An error occurred: {error_msg}"
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return

        summary_text = "".join(full_summary_parts)

        # ── Post-process: enforce word limit for paragraphs mode ──────────
        if req.format_type == "paragraphs":
            summary_text = enforce_word_limit(summary_text, req.length)
            # Send a correction token if we trimmed anything
            yield f"data: {json.dumps({'token': None, 'corrected_summary': summary_text})}\n\n"

        # ── Stage 3: Extract action items + blockers ──────────────────────
        action_items, blockers = [], []
        try:
            ai_resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=ACTION_PROMPT.format(transcript=cleaned),
            )
            raw = re.sub(r"^```[a-z]*\n?|\n?```$", "", ai_resp.text.strip())
            parsed = json.loads(raw)
            action_items = parsed.get("action_items", [])
            blockers     = parsed.get("blockers", [])
        except Exception as e:
            print(f"Action items extraction failed: {e}")

        yield f"event: action_items\ndata: {json.dumps({'action_items': action_items, 'blockers': blockers})}\n\n"

        # ── Stage 4: Save to SQLite ───────────────────────────────────────
        record_id = None
        try:
            conn = get_db(project)
            cur = conn.execute(
                """INSERT INTO standups
                   (created_at, raw_notes, summary, format_type,
                    action_items, blockers, members, project)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    req.text, summary_text, req.format_type,
                    json.dumps(action_items), json.dumps(blockers),
                    json.dumps(names), project,
                ),
            )
            conn.commit()
            record_id = cur.lastrowid
            conn.close()
        except Exception as e:
            print(f"DB save failed: {e}")

        yield f"event: meta\ndata: {json.dumps({'project': project, 'names': names, 'record_id': record_id})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history/{project}")
def get_history(project: str, limit: int = 20):
    try:
        conn = get_db(project)
        rows = conn.execute(
            "SELECT * FROM standups WHERE project=? ORDER BY id DESC LIMIT ?",
            (project, limit),
        ).fetchall()
        conn.close()
        return {"project": project, "records": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects")
def list_projects():
    return {"projects": sorted(p.stem for p in DATA_DIR.glob("*.db"))}