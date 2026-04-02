from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Standup Summarizer API")

class SummaryRequest(BaseModel):
    text: str
    length: int = 50
    format_type: str = "paragraph"
    slide_count: Optional[int] = 0

@app.get("/")
def home():
    return {"message":"My FastAPI Backend is Alive"}

@app.post("/summary")

def summarize_text(request: SummaryRequest):
    mock_response = f"Received your text. You want a {request.length}% length summary formatted as {request.format_type}"

    if request.format_type == "slides":
        mock_response+= f"You asked for {request.slide_count} slides"

    return {"summary":mock_response}
