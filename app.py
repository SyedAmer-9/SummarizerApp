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

KNOWN_PROJECTS = ["onblick", "reccopilot", "sales", "rig", "hrbp", "general"]


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
    Hard-truncate to `limit` words, cutting at the last complete sentence.
    Only applied in paragraphs mode.
    """
    words = text.split()
    if len(words) <= limit:
        return text
    truncated = " ".join(words[:limit])
    match = re.search(r'(.*[.!?])\s', truncated + " ", re.DOTALL)
    return match.group(1).strip() if match else truncated.strip()


# ── Transcript parser ──────────────────────────────────────────────────────
def parse_and_clean_transcript(text: str) -> tuple[str, list[str]]:
    """
    Handles three common Teams transcript formats:

    Format A — inline:
        Alice Kumar: I finished the auth module.

    Format B — name/time on separate lines (most common Teams export):
        Alice Kumar
        10:05 AM
        I finished the auth module.

    Format C — bracketed timestamp prefix:
        [10:05 AM] Alice Kumar: I finished the auth module.

    Returns (cleaned_text, sorted_list_of_unique_first_names).
    """

    NOISE_PHRASES = [
        "joined the meeting", "left the meeting", "raised hand",
        "lowered hand", "can you hear me", "let me share",
        "you're on mute", "start recording", "stopped recording",
        "meeting started", "meeting ended", "turned on",
        "turned off", "background noise", "transcript",
    ]

    # Words that look capitalised but are NOT person names
    STOPWORDS = {
        "note", "meeting", "everyone", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday", "today", "tomorrow",
        "hi", "hey", "okay", "thanks", "thank", "sure", "yes", "no",
        "teams", "sharepoint", "microsoft", "onblick", "reccopilot",
        "sales", "rig", "hrbp", "general", "unmute", "mute",
    }

    lines = [l.strip() for l in text.split("\n")]
    names: set[str] = set()
    cleaned: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip blank lines and noise
        if not line:
            i += 1
            continue
        if any(noise in line.lower() for noise in NOISE_PHRASES):
            i += 1
            continue

        # ── Format B detection ─────────────────────────────────────────
        # Pattern: a plain name line (no colon) followed by a timestamp line,
        # followed by one or more content lines.
        # e.g.  "Alice Kumar"  /  "10:05 AM"  /  "I finished the PR."
        is_name_line = (
            re.match(r'^[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3}$', line)
            and ":" not in line
            and len(line.split()) <= 4
        )
        next_is_timestamp = (
            i + 1 < len(lines)
            and re.match(r'^\d{1,2}:\d{2}\s*(?:AM|PM)?$', lines[i + 1].strip(), re.IGNORECASE)
        )

        if is_name_line and next_is_timestamp:
            speaker = line
            first_name = speaker.split()[0]
            if first_name.lower() not in STOPWORDS:
                names.add(first_name)
            # Skip the timestamp line
            i += 2
            # Collect all content lines until the next speaker block or blank
            content_lines = []
            while i < len(lines):
                next_line = lines[i].strip()
                # Stop if we hit another name+timestamp block
                if (
                    re.match(r'^[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3}$', next_line)
                    and ":" not in next_line
                    and len(next_line.split()) <= 4
                    and i + 1 < len(lines)
                    and re.match(r'^\d{1,2}:\d{2}\s*(?:AM|PM)?$', lines[i + 1].strip(), re.IGNORECASE)
                ):
                    break
                if next_line:
                    content_lines.append(next_line)
                i += 1
            if content_lines:
                cleaned.append(f"{speaker}: {' '.join(content_lines)}")
            continue

        # ── Format A / Format C ────────────────────────────────────────
        # Matches "Alice:" or "[10:05 AM] Alice Kumar:" or "Alice Kumar:"
        match = re.match(r'^(?:\[.*?\]\s*)?([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3})\s*:', line)
        if match:
            speaker = match.group(1).strip()
            first_name = speaker.split()[0]
            if (
                first_name.lower() not in STOPWORDS
                and len(speaker) < 40
            ):
                names.add(first_name)
            cleaned.append(line)
        else:
            # Keep lines that aren't noise and aren't bare metadata
            # Filter out lone timestamps and short filler
            if len(line) > 5 and not re.match(r'^\d{1,2}:\d{2}', line):
                cleaned.append(line)

        i += 1

    return "\n".join(cleaned), sorted(names)


# ── Pydantic model ─────────────────────────────────────────────────────────
class SummaryRequest(BaseModel):
    text: str
    length: int = 50
    format_type: str = "paragraphs"
    slide_count: Optional[int] = 3
    points_count: Optional[int] = 5
    project: str  # mandatory — user always selects explicitly


# ── Prompt builders ────────────────────────────────────────────────────────
def build_summary_prompt(req: SummaryRequest, names: list[str], cleaned: str) -> str:
    name_section = ""
    if names:
        name_section = (
            f"These people participated: {', '.join(names)}.\n"
            "Group the summary by person — one section per speaker.\n\n"
        )

    if req.format_type == "points":
        fmt = (
            f"Format: exactly {req.points_count} bullet points using Markdown (- item).\n"
            f"Hard limit: {req.points_count} bullets. No more, no fewer.\n"
            "Each bullet is one concise sentence."
        )
    elif req.format_type == "slides":
        fmt = (
            f"Format: exactly {req.slide_count} slides separated by --- (Markdown HR).\n"
            f"Hard limit: exactly {req.slide_count} slides.\n"
            "Each slide needs a ## heading and at least 50 words of content."
        )
    else:
        fmt = (
            f"Format: clear paragraphs in Markdown.\n"
            f"HARD WORD LIMIT: {req.length} words MAXIMUM. "
            f"This is a strict ceiling — plan your response before writing and do not exceed it."
        )

    return (
        "You are an expert technical scribe for a software company. "
        "Summarize the following standup transcript accurately and concisely.\n\n"
        f"{name_section}"
        f"{fmt}\n\n"
        f"Transcript:\n{cleaned}"
    )


ACTION_PROMPT = """\
Extract structured information from this standup transcript.

Return ONLY valid JSON — no markdown fences, no explanation:
{{"action_items": ["...", "..."], "blockers": ["...", "..."]}}

Rules:
- action_items: concrete tasks someone explicitly committed to doing
- blockers: things actively slowing the team down right now
- One sentence per item
- Attribute to a person where possible, e.g. "Alice to review PR #42"
- Empty array if nothing found for a category

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
    SSE stream. Events emitted in order:
      data: {"stage": "cleaning"}
      data: {"stage": "names", "names": [...]}
      data: {"token": "..."}                         ← repeated until summary complete
      data: {"token": null, "corrected_summary":"…"} ← only if word limit was exceeded
      event: action_items  data: {"action_items":[], "blockers":[]}
      event: meta          data: {"project":"…", "names":[], "record_id": N}
      event: done          data: {}
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    if req.project not in KNOWN_PROJECTS:
        raise HTTPException(status_code=400, detail=f"Unknown project '{req.project}'.")

    async def stream():

        # ── Stage 0 & 1: Instant Python parsing (no API call needed) ──────
        yield f"data: {json.dumps({'stage': 'cleaning'})}\n\n"
        cleaned, names = parse_and_clean_transcript(req.text)
        yield f"data: {json.dumps({'stage': 'names', 'names': names})}\n\n"

        # ── Stage 2: Stream the summary ────────────────────────────────────
        summary_prompt = build_summary_prompt(req, names, cleaned)
        full_parts: list[str] = []

        try:
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=summary_prompt,
            ):
                token = chunk.text or ""
                full_parts.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                friendly = "Rate limit reached — please wait 10 seconds and try again."
            elif "API_KEY" in error_msg.upper():
                friendly = "API key error — check your .env file."
            else:
                friendly = f"Gemini error: {error_msg}"
            # BUG FIX: was building `user_msg` but yielding raw `str(e)` — now yields friendly message
            yield f"event: error\ndata: {json.dumps({'error': friendly})}\n\n"
            return

        summary_text = "".join(full_parts)

        # ── Post-process: hard word-limit enforcement for paragraphs ───────
        if req.format_type == "paragraphs":
            enforced = enforce_word_limit(summary_text, req.length)
            if enforced != summary_text:
                # Only send correction event if we actually trimmed something
                summary_text = enforced
                yield f"data: {json.dumps({'token': None, 'corrected_summary': summary_text})}\n\n"

        # ── Stage 3: Action items + blockers ───────────────────────────────
        action_items: list[str] = []
        blockers: list[str] = []
        try:
            ai_resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=ACTION_PROMPT.format(transcript=cleaned),
            )
            raw = re.sub(r"^```[a-z]*\n?|\n?```$", "", ai_resp.text.strip())
            parsed = json.loads(raw)
            action_items = parsed.get("action_items", [])
            blockers = parsed.get("blockers", [])
        except Exception as e:
            print(f"Action items extraction failed: {e}")

        yield f"event: action_items\ndata: {json.dumps({'action_items': action_items, 'blockers': blockers})}\n\n"

        # ── Stage 4: Persist to SQLite ─────────────────────────────────────
        record_id = None
        try:
            conn = get_db(req.project)
            cur = conn.execute(
                """INSERT INTO standups
                   (created_at, raw_notes, summary, format_type,
                    action_items, blockers, members, project)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    req.text, summary_text, req.format_type,
                    json.dumps(action_items), json.dumps(blockers),
                    json.dumps(names), req.project,
                ),
            )
            conn.commit()
            record_id = cur.lastrowid
            conn.close()
        except Exception as e:
            print(f"DB save failed: {e}")

        yield f"event: meta\ndata: {json.dumps({'project': req.project, 'names': names, 'record_id': record_id})}\n\n"
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