import os
import io
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
async def health():
    return {"status": "AgreeLens backend running"}

# -----------------------------
# Extraction Functions
# -----------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print("PDF extraction error:", e)
    return text

def extract_text_from_docx(file_bytes):
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in document.paragraphs])
    except Exception as e:
        print("DOCX extraction error:", e)
        return ""

# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are AgreeLens — an educational document clarity assistant.

IMPORTANT:
- Educational explanations only.
- No legal advice.
- Always return structured JSON only.

Return JSON with:

{
  "quick_summary": "...",
  "detailed_summary": "...",
  "key_takeaways": [],
  "obligations": [],
  "important_dates": [],
  "risks_or_flags": [],
  "educational_notes": [],
  "improvement_suggestions_general": [],
  "expert_review_recommended": false,
  "reason_for_expert_review": "",
  "analysis_feedback": ""
}
"""

# -----------------------------
# ANALYZE ENDPOINT
# -----------------------------
@app.post("/analyze")
async def analyze_document(payload: dict = Body(...)):

    try:
        file_base64 = payload.get("file_base64")
        file_name = payload.get("file_name", "")

        if not file_base64:
            return {"error": "No file data received."}

        if "," in file_base64:
            file_base64 = file_base64.split(",")[1]

        try:
            file_bytes = base64.b64decode(file_base64)
        except Exception:
            return {"error": "Invalid base64 format."}

        # Detect file type
        if file_name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif file_name.lower().endswith(".docx"):
            text = extract_text_from_docx(file_bytes)
        else:
            return {"error": "Unsupported file type."}

        if not text.strip():
            return {"error": "No readable text found in document."}

        text = text[:8000]

        # Call OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.2
            )

            analysis_text = response.choices[0].message.content

        except Exception as e:
            print("OpenAI error:", e)
            return {"error": "AI processing failed."}

        # Parse JSON safely
        try:
            result = json.loads(analysis_text)
        except Exception:
            print("Invalid JSON from model:", analysis_text)
            return {"error": "Model returned invalid JSON."}

        # 🔒 HARD SAFETY NORMALIZATION
        result.setdefault("quick_summary", "")
        result.setdefault("detailed_summary", "")
        result.setdefault("key_takeaways", [])
        result.setdefault("obligations", [])
        result.setdefault("important_dates", [])
        result.setdefault("risks_or_flags", [])
        result.setdefault("educational_notes", [])
        result.setdefault("improvement_suggestions_general", [])
        result.setdefault("expert_review_recommended", False)
        result.setdefault("reason_for_expert_review", "")
        result.setdefault("analysis_feedback", "")

        # Ensure obligations always safe
        safe_obligations = []
        for obligation in result.get("obligations", []):
            if isinstance(obligation, dict):
                description = obligation.get("description") or ""
                obligation["text_en"] = str(description)
                safe_obligations.append(obligation)

        result["obligations"] = safe_obligations

        return result

    except Exception as e:
        print("Unexpected error:", e)
        return {"error": "Internal server error."}