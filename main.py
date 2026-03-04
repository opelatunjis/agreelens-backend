import os
import io
import json
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, Form
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


# -------- MAIN ANALYSIS ENDPOINT -------- #

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    document_type: str = Form(None),
    user_role: str = Form(None)
):
    contents = await file.read()

    # Detect file type
    if file.filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(contents)
    elif file.filename.lower().endswith(".docx"):
        text = extract_text_from_docx(contents)
    else:
        return {"error": "Unsupported file type. Please upload PDF or DOCX."}

    if not text.strip():
        return {"error": "No readable text found in document."}

    # Safety limit
    text = text[:12000]

    SYSTEM_PROMPT = f"""
You are AgreeLens.

You are an educational document clarity assistant.
You are NOT a lawyer and must NOT provide legal advice.

Context:
Document Type: {document_type}
User Role: {user_role}

Return ONLY valid JSON in this structure:

{{
  "key_highlights": [],
  "obligations": [],
  "risks": [],
  "action_items": [],
  "deadlines": [],
  "financial_exposure": [],
  "definitions": [],
  "escalation_recommended": false
}}

Escalation rule:
Set escalation_recommended = true ONLY if:
- High financial liability
- Personal guarantees
- Indemnification clauses
- Termination penalties
- Litigation or legal threat language
- Complex regulatory exposure

Rules:
- Be factual.
- Be concise.
- No exaggeration.
- No legal advice.
- Populate all fields (empty list if none).
"""

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
        return json.loads(analysis_text)
    except Exception:
        return {
            "error": "Model did not return valid JSON",
            "raw_response": analysis_text
        }