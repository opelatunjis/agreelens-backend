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

# Load environment variables
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

# -------- TEXT EXTRACTION FUNCTIONS -------- #

def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def extract_text_from_docx(file_bytes):
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in document.paragraphs])


# -------- SYSTEM PROMPT -------- #

SYSTEM_PROMPT = """
You are AgreeLens.

You are an educational document clarity assistant.
You are NOT a lawyer and must NOT provide legal advice.

Return ONLY valid JSON in this structure:

{
  "key_highlights": [],
  "obligations": [],
  "risks": [],
  "action_items": [],
  "deadlines": [],
  "financial_exposure": [],
  "definitions": [],
  "escalation_recommended": false
}

Rules:
- Be factual.
- Be concise.
- No exaggeration.
- No legal advice.
- Populate all fields (empty list if none).
"""

# -------- MAIN ANALYSIS ENDPOINT -------- #

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

    if not text.strip():
        return {"error": "No readable text found in document."}

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
        return json.loads(analysis_text)

    except Exception as e:
        print("OpenAI error:", e)
        return {"error": "AI processing failed."}