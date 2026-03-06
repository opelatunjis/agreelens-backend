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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Health Check Route
# -----------------------------
@app.get("/")
async def health():
    return {"status": "AgreeLens backend running"}


# -----------------------------
# Text Extraction Functions
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
        return "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        print("DOCX extraction error:", e)
        return ""


# -----------------------------
# System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are AgreeLens — an educational document clarity assistant.

IMPORTANT RULES:
- This tool provides educational and informational explanations only.
- Do NOT provide legal advice.
- Do NOT instruct the user what they should do.
- Do NOT interpret enforceability.
- If information is unclear, say: "Not explicitly stated."
- Always return structured JSON only.

Your goal:
Help the reader understand what the document says in clear, neutral language.

Analyze the extracted text and return JSON with the following structure:

{
  "quick_summary": "5-8 sentence plain-language overview of what this document is about.",
  "detailed_summary": "Clear explanation of important clauses and what they mean in practical terms.",
  "key_takeaways": ["bullet points of main points"],
  "obligations": [
    {
      "description": "What the party must do",
      "responsible_party": "Who is responsible",
      "due_date": "Date if mentioned or 'Not explicitly stated'",
      "reference_excerpt": "Short excerpt from document supporting this",
      "reference_location": "Section name or paragraph number if available"
    }
  ],
  "important_dates": [
    {
      "date": "Date mentioned",
      "context": "What happens on this date",
      "reference_excerpt": "Supporting excerpt"
    }
  ],
  "risks_or_flags": [
    {
      "description": "Neutral explanation of potential concern",
      "confidence": "Low/Medium/High",
      "reference_excerpt": "Supporting excerpt"
    }
  ],
  "educational_notes": [
    {
      "term": "Legal or technical term",
      "plain_explanation": "Simple educational explanation"
    }
  ],
  "improvement_suggestions_general": [
    "General best-practice suggestion based on common contract standards (NOT specific legal advice)."
  ],
  "expert_review_recommended": true_or_false,
  "reason_for_expert_review": "If true, explain why in neutral language.",
  "analysis_feedback": "Short self-evaluation of analysis confidence."
}

Guidelines:
- Base analysis ONLY on provided text.
- Extract exact dates and values when possible.
- If section numbers exist, include them.
- If no section number, say: "Paragraph location approximate."
- Keep explanations neutral and educational.
"""

# -----------------------------
# Main Analysis Endpoint
# -----------------------------
@app.post("/analyze")
async def analyze_document(payload: dict = Body(...)):

    file_base64 = payload.get("file_base64")
    file_name = payload.get("file_name", "")

    if not file_base64:
        return {"error": "No file data received."}

    # Remove data URL prefix if present
    if "," in file_base64:
        file_base64 = file_base64.split(",")[1]

    try:
        file_bytes = base64.b64decode(file_base64)
    except Exception as e:
        print("Base64 decode error:", e)
        return {"error": "Invalid base64 file format."}

    # Detect file type
    if file_name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif file_name.lower().endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    else:
        return {"error": "Unsupported file type."}

    print("Extracted text length:", len(text))

    if not text.strip():
        return {"error": "No readable text found in document."}

    # Limit text size for performance
    text = text[:8000]

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

# Ensure valid JSON response
try:
    result = json.loads(analysis_text)

    # 🔒 Extra safety compatibility patch
    for obligation in result.get("obligations", []):
        description = obligation.get("description") or ""
        obligation["text_en"] = str(description)

    return result

except Exception:
    print("Model returned invalid JSON:", analysis_text)
    return {
        "error": "Model did not return valid JSON",
        "raw_response": analysis_text
    }

except Exception:
    print("Model returned invalid JSON:", analysis_text)
    return {
        "error": "Model did not return valid JSON",
        "raw_response": analysis_text
    }

    except Exception as e:
        print("OpenAI error:", e)
        return {"error": "AI processing failed."}