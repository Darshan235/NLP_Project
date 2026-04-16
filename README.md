# 🚀 NLP Processor API

An AI-powered FastAPI application that generates **summaries, key sentences, and MCQs** from input text using **Hugging Face Transformers (FLAN-T5)**.

---

## 📌 Features

- 📝 Automatic text summarization  
- 🔍 Key sentence extraction  
- ❓ AI-generated questions  
- 🧠 MCQ generation with options  
- ⚡ FastAPI backend with interactive docs  

---

## 🛠️ Tech Stack

- **Backend:** FastAPI  
- **ML/NLP:** Hugging Face Transformers (FLAN-T5)  
- **Language:** Python  
- **Other:** NLTK  

---

## 📂 Project Structure


NLP_Project/
│── NLP/
│ └── app.py
│── index.html (optional frontend)
│── README.md


---

## ⚙️ Installation

### 1. Clone the repository


git clone https://github.com/your-username/nlp-processor-api.git

cd nlp-processor-api


### 2. Install dependencies


pip install fastapi uvicorn transformers torch nltk


### 3. Download NLTK data (one-time setup)


python -c "import nltk; nltk.download('punkt')"


---

## ▶️ Running the Application


uvicorn NLP.app:app --reload


---

## 🌐 API Endpoints

### 🔹 Home

GET /


### 🔹 Process Text

POST /process


#### Request Body:
```json
{
  "text": "Your input text here"
}
Response:
{
  "notes": "...",
  "key_sentences": ["...", "..."],
  "mcqs": [
    {
      "question": "...",
      "answer": "...",
      "options": ["...", "...", "...", "..."]
    }
  ]
}
🧪 API Testing

Open in browser:

http://127.0.0.1:8000/docs

Use Swagger UI to test endpoints.


