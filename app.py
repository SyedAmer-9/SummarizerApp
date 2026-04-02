import os
from fastapi import FastAPI, Request,HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please check your .env file.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')


app = FastAPI(title="Standup Summarizer API")
templates = Jinja2Templates(directory="templates")

class SummaryRequest(BaseModel):
    text: str
    length: int = 50
    format_type: str = "paragraph"
    slide_count: Optional[int] = 0
    points_count: Optional[int] = 5

@app.get("/")
def home(request:Request):
    return templates.TemplateResponse(request=request,name="index.html")

@app.post("/summarize")

def summarize_text(request: SummaryRequest):
    try:
        system_instruction = f"You are an expert technical assistant. Summarize the following standup notes. "
        system_instruction += f"Strictly limit the total summary to around {request.length} words. "
        
        if request.format_type == "points":
            system_instruction += f"Format the output as exactly {request.points_count} clean, professional bullet points using Markdown."        
        elif request.format_type == "slides":
            system_instruction += f"Format the output as a presentation with exactly {request.slide_count} slides. Separate each slide with a Markdown horizontal rule (---) and use line breaks so it is easy to read."
        else:
            system_instruction += "Format the output in clear, readable paragraphs using Markdown."
        
        final_prompt = f"{system_instruction}\n\nHere are the notes to summarize:\n{request.text}"
        
        response = model.generate_content(final_prompt)

        return {"summary":response.text}
    
    except Exception as e:
        # If anything goes wrong (e.g., API is down, text is too long), tell the frontend safely
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary.")
         


