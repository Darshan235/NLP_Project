from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import random
import os

# ---------- NLTK setup ----------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ---------- FastAPI app ----------
app = FastAPI(title="NLP Processor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- IMPORTANT PATH FIX ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- Model ----------
MODEL_NAME = "google/flan-t5-small"

# ---------- Lazy loading ----------
summarizer_tokenizer = None
summarizer_model = None
qg_tokenizer = None
qg_model = None

def load_models():
    global summarizer_tokenizer, summarizer_model, qg_tokenizer, qg_model

    if summarizer_model is None:
        summarizer_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if qg_model is None:
        qg_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        qg_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ---------- Request schema ----------
class ProcessRequest(BaseModel):
    text: str

# ---------- Core logic ----------
def generate_notes(text: str) -> str:
    chunks = [text[i:i + 800] for i in range(0, len(text), 800)]
    notes = ""

    for chunk in chunks:
        prompt = "summarize: " + chunk
        inputs = summarizer_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        summary_ids = summarizer_model.generate(
            inputs.input_ids,
            max_length=120,
            min_length=30,
            num_beams=2
        )

        summary = summarizer_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        notes += summary.strip() + "\n"

    return notes.strip()


def extract_key_sentences(text: str, n: int = 5):
    sentences = sent_tokenize(text)
    return sentences[:n]


def generate_questions(sentences):
    questions = []

    for sentence in sentences:
        prompt = "generate question: " + sentence
        inputs = qg_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        question_ids = qg_model.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=2
        )

        question = qg_tokenizer.decode(
            question_ids[0],
            skip_special_tokens=True
        ).strip()

        questions.append({
            "question": question,
            "answer": sentence
        })

    return questions


def add_options(mcqs):
    all_answers = [m["answer"] for m in mcqs]

    for mcq in mcqs:
        options = random.sample(all_answers, min(4, len(all_answers)))

        if mcq["answer"] not in options:
            options[random.randint(0, len(options) - 1)] = mcq["answer"]

        random.shuffle(options)
        mcq["options"] = options

    return mcqs


def process_text(text: str):
    if not text.strip():
        raise ValueError("Input text is empty.")

    load_models()

    notes = generate_notes(text)
    key_sentences = extract_key_sentences(text)
    mcqs = generate_questions(key_sentences)
    mcqs = add_options(mcqs)

    return {
        "notes": notes,
        "key_sentences": key_sentences,
        "mcqs": mcqs
    }

# ---------- ROUTES ----------

@app.get("/")
def home():
    file_path = os.path.join(BASE_DIR, "home.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"detail": "home.html not found"})


@app.get("/app")
def app_page():
    file_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"detail": "index.html not found"})


@app.post("/process")
def process(request: ProcessRequest):
    try:
        return process_text(request.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))