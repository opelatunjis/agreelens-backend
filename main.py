import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
import io
from openai import OpenAI

app = FastAPI()

# Allow frontend to connect later (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 Replace this with your real OpenAI API key
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
async def analyze_document(file: UploadFile = File(...)):
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

    # Limit size for safety
    text = text[:12000]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are AgreeLens Beta. Analyze this document clearly and return a structured summary including key obligations, risks, deadlines, and important clauses."
            },
            {"role": "user", "content": text}
        ],
        temperature=0.2
    )

    return {
        "analysis": response.choices[0].message.content
    }