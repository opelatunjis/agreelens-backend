import os
import io
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    text = text[:12000]

    SYSTEM_PROMPT = f"""
You are AgreeLens.

You are an educational clarity assistant.
You are NOT a lawyer and must NOT provide legal advice.

Context:
Document Type: {document_type}
User Role: {user_role}

Return ONLY valid JSON:

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

Be factual.
Be concise.
No exaggeration.
No legal advice.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content