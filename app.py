import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please check your .env file.")

client = genai.Client(api_key=api_key)

app = FastAPI(title="Standup Summarizer API")
templates = Jinja2Templates(directory="templates")


class SummaryRequest(BaseModel):
    text: str
    length: int = 50
    format_type: str = "paragraph"
    slide_count: Optional[int] = 0
    points_count: Optional[int] = 5


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ── NEW: Health check endpoint ──
# The frontend calls GET /health every 30 seconds.
# We do a lightweight Gemini API probe (count_tokens is free & fast).
# Returns 200 if the API key is valid and reachable, 503 otherwise.
@app.get("/health")
def health_check():
    try:
        client.models.count_tokens(
            model="gemini-2.5-flash",
            contents="ping"
        )
        return {"status": "ok"}
    except Exception as e:
        print(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Gemini API unreachable")


@app.post("/summarize")
def summarize_text(request: SummaryRequest):
    try:
        system_instruction = "You are an expert technical assistant. Summarize the following standup notes. "

        if request.format_type == "points":
            system_instruction += f"Format the output as exactly {request.points_count} clean, professional bullet points using Markdown."
        elif request.format_type == "slides":
            system_instruction += (
                f"Format the output as a presentation with exactly {request.slide_count} slides. "
                f"Separate each slide with a Markdown horizontal rule (---) and use line breaks so it is easy to read. "
                f"CRITICAL REQUIREMENT: Every single slide MUST contain a minimum of 50 words. "
                f"If the provided notes lack sufficient detail to reach 50 words per slide, "
                f"you must elaborate by adding relevant basic definitions, industry context, or logical improvisations "
                f"based on the original topic to ensure the length requirement is met."
            )
        else:
            system_instruction += f"Format the output in clear, readable paragraphs using Markdown. Strictly limit the total summary to around {request.length} words."

        final_prompt = f"{system_instruction}\n\nHere are the notes to summarize:\n{request.text}"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt
        )

        return {"summary": response.text}

    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary.")