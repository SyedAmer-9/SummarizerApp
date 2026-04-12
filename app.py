import os
import re
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

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
    words = text.split()
    if len(words) <= limit:
        return text
    truncated = " ".join(words[:limit])
    match = re.search(r'(.*[.!?])\s', truncated + " ", re.DOTALL)
    return match.group(1).strip() if match else truncated.strip()


# ── Transcript parser ──────────────────────────────────────────────────────
def parse_and_clean_transcript(text: str) -> tuple[str, list[str]]:
    NOISE_PHRASES = [
        "joined the meeting", "left the meeting", "raised hand",
        "lowered hand", "can you hear me", "let me share",
        "you're on mute", "start recording", "stopped recording",
        "meeting started", "meeting ended", "turned on",
        "turned off", "background noise", "transcript",
    ]
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
        if not line:
            i += 1
            continue
        if any(noise in line.lower() for noise in NOISE_PHRASES):
            i += 1
            continue

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
            i += 2
            content_lines = []
            while i < len(lines):
                next_line = lines[i].strip()
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

        match = re.match(r'^(?:\[.*?\]\s*)?([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3})\s*:', line)
        if match:
            speaker = match.group(1).strip()
            first_name = speaker.split()[0]
            if first_name.lower() not in STOPWORDS and len(speaker) < 40:
                names.add(first_name)
            cleaned.append(line)
        else:
            if len(line) > 5 and not re.match(r'^\d{1,2}:\d{2}', line):
                cleaned.append(line)
        i += 1

    return "\n".join(cleaned), sorted(names)


# ── Pydantic models ────────────────────────────────────────────────────────
class SummaryRequest(BaseModel):
    text: str
    length: int = 50
    format_type: str = "paragraphs"
    slide_count: Optional[int] = 3
    points_count: Optional[int] = 5
    project: str


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
            "This is a strict ceiling — plan your response before writing and do not exceed it."
        )
    return (
        "You are an expert technical scribe for a software company. "
        "Summarize the following standup transcript accurately and concisely.\n\n"
        f"{name_section}{fmt}\n\nTranscript:\n{cleaned}"
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


# ── Insight helpers ────────────────────────────────────────────────────────
def get_week_bounds(offset_weeks: int = 0):
    """Return (start, end) ISO strings for a week, 0 = this week, -1 = last week."""
    today = datetime.utcnow().date()
    start_of_week = today - timedelta(days=today.weekday()) - timedelta(weeks=-offset_weeks)
    end_of_week   = start_of_week + timedelta(days=6)
    return start_of_week.isoformat(), end_of_week.isoformat()


def words_overlap(a: str, b: str) -> float:
    """Return Jaccard similarity between two strings (word sets)."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def detect_carryover(this_items: list[str], last_items: list[str], threshold: float = 0.5) -> list[str]:
    """Return items from this_items that look like repeats of something in last_items."""
    carried = []
    for item in this_items:
        for prev in last_items:
            if words_overlap(item, prev) >= threshold:
                carried.append(item)
                break
    return carried


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/history-ui")
def history_ui(request: Request):
    return templates.TemplateResponse(request=request, name="history.html")


@app.get("/insights-ui")
def insights_ui(request: Request):
    return templates.TemplateResponse(request=request, name="insights.html")


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
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")
    if req.project not in KNOWN_PROJECTS:
        raise HTTPException(status_code=400, detail=f"Unknown project '{req.project}'.")

    async def stream():
        yield f"data: {json.dumps({'stage': 'cleaning'})}\n\n"
        cleaned, names = parse_and_clean_transcript(req.text)
        yield f"data: {json.dumps({'stage': 'names', 'names': names})}\n\n"

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
            yield f"event: error\ndata: {json.dumps({'error': friendly})}\n\n"
            return

        summary_text = "".join(full_parts)

        if req.format_type == "paragraphs":
            enforced = enforce_word_limit(summary_text, req.length)
            if enforced != summary_text:
                summary_text = enforced
                yield f"data: {json.dumps({'token': None, 'corrected_summary': summary_text})}\n\n"

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
def get_history(project: str, limit: int = 30):
    """Return last N standup records for a project, with parsed JSON fields."""
    try:
        conn = get_db(project)
        rows = conn.execute(
            "SELECT * FROM standups WHERE project=? ORDER BY id DESC LIMIT ?",
            (project, limit),
        ).fetchall()
        conn.close()
        records = []
        for r in rows:
            d = dict(r)
            for field in ("action_items", "blockers", "members"):
                try:
                    d[field] = json.loads(d[field] or "[]")
                except Exception:
                    d[field] = []
            records.append(d)
        return {"project": project, "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects")
def list_projects():
    return {"projects": sorted(p.stem for p in DATA_DIR.glob("*.db"))}


@app.get("/insights/{project}")
def get_insights(project: str):
    """
    Returns week-over-week comparison data for a project.
    No LLM calls — pure SQL + Python math.
    """
    try:
        conn = get_db(project)

        this_start, this_end   = get_week_bounds(0)
        last_start, last_end   = get_week_bounds(-1)

        def fetch_week(start: str, end: str):
            return conn.execute(
                """SELECT * FROM standups
                   WHERE project=? AND date(created_at) BETWEEN ? AND ?
                   ORDER BY created_at ASC""",
                (project, start, end),
            ).fetchall()

        this_rows = fetch_week(this_start, this_end)
        last_rows = fetch_week(last_start, last_end)
        conn.close()

        def aggregate(rows):
            all_action_items, all_blockers, all_members = [], [], []
            daily_counts = defaultdict(int)  # date → standup count
            member_counts = defaultdict(int) # name → appearances

            for r in rows:
                day = r["created_at"][:10]
                daily_counts[day] += 1
                try:
                    ais = json.loads(r["action_items"] or "[]")
                    all_action_items.extend(ais)
                except Exception:
                    pass
                try:
                    bls = json.loads(r["blockers"] or "[]")
                    all_blockers.extend(bls)
                except Exception:
                    pass
                try:
                    mems = json.loads(r["members"] or "[]")
                    all_members.extend(mems)
                    for m in mems:
                        member_counts[m] += 1
                except Exception:
                    pass

            # Blocker frequency
            blocker_freq: dict[str, int] = defaultdict(int)
            for b in all_blockers:
                blocker_freq[b] += 1

            return {
                "standup_count":  len(rows),
                "action_item_count": len(all_action_items),
                "blocker_count":  len(all_blockers),
                "unique_members": len(set(all_members)),
                "daily_counts":   dict(daily_counts),
                "member_counts":  dict(member_counts),
                "top_blockers":   sorted(blocker_freq.items(), key=lambda x: -x[1])[:5],
                "all_action_items": all_action_items,
                "all_blockers":   all_blockers,
            }

        this_agg = aggregate(this_rows)
        last_agg = aggregate(last_rows)

        # Carry-over detection: action items from this week that also appeared last week
        carryover = detect_carryover(
            this_agg["all_action_items"],
            last_agg["all_action_items"],
        )

        # Recurring blockers: appear in both weeks
        recurring_blockers = detect_carryover(
            this_agg["all_blockers"],
            last_agg["all_blockers"],
            threshold=0.4,
        )

        # Build chart data: last 14 days with standup counts per day
        chart_days = []
        for i in range(13, -1, -1):
            d = (datetime.utcnow().date() - timedelta(days=i)).isoformat()
            this_c = this_agg["daily_counts"].get(d, 0)
            last_c = last_agg["daily_counts"].get(d, 0)
            chart_days.append({"date": d, "this_week": this_c, "last_week": last_c})

        def pct_change(now, prev):
            if prev == 0:
                return None
            return round(((now - prev) / prev) * 100, 1)

        return {
            "project": project,
            "this_week": {
                "range": f"{this_start} → {this_end}",
                **this_agg,
            },
            "last_week": {
                "range": f"{last_start} → {last_end}",
                **last_agg,
            },
            "changes": {
                "standup_count":     pct_change(this_agg["standup_count"],     last_agg["standup_count"]),
                "action_item_count": pct_change(this_agg["action_item_count"], last_agg["action_item_count"]),
                "blocker_count":     pct_change(this_agg["blocker_count"],     last_agg["blocker_count"]),
            },
            "carryover_action_items": carryover,
            "recurring_blockers":     recurring_blockers,
            "chart_data":             chart_days,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))