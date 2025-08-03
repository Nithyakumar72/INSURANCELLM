from flask import Flask, request, jsonify
import os
import requests
import PyPDF2
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import time

app = Flask(__name__)

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment")
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini model
gemini_model = genai.GenerativeModel("models/gemini-pro")

# Load SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_clauses_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    clauses = [line.strip() for line in text.split('\n') if line.strip()]
    return clauses


def create_faiss_index(clauses):
    embeddings = embedding_model.encode(clauses)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index


def get_top_k_clauses(question, clauses, index, k=3):
    question_embedding = embedding_model.encode([question])
    _, I = index.search(np.array(question_embedding), k)
    return [clauses[i] for i in I[0]]


@app.route("/hackrx/run", methods=["POST"])
def run():
    try:
        start_time = time.time()

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        user_api_key = auth_header.replace("Bearer ", "").strip()
        valid_api_key = os.getenv("HACKRX_API_KEY", "d464d88731074c5923019b6139916b4ba2e7cd1b8cb01316fed78295b75c066e")
        if user_api_key != valid_api_key:
            return jsonify({'error': 'Unauthorized API key'}), 403

        data = request.get_json()
        pdf_url = data.get("documents")
        questions = data.get("questions", [])

        if not pdf_url or not questions:
            return jsonify({'error': 'Missing PDF URL or questions'}), 400

        pdf_response = requests.get(pdf_url, timeout=20)
        if pdf_response.status_code != 200:
            return jsonify({'error': f'Failed to download PDF from {pdf_url}'}), 400

        with open("temp.pdf", "wb") as f:
            f.write(pdf_response.content)

        clauses = extract_clauses_from_pdf("temp.pdf")
        if not clauses:
            return jsonify({'error': 'No clauses extracted from PDF'}), 400

        index = create_faiss_index(clauses)

        answers = []
        for question in questions:
            top_clauses = get_top_k_clauses(question, clauses, index)
            prompt = f"""You are a helpful assistant for insurance policy documents.
Answer the question based **only** on the relevant policy clauses below. Be direct, accurate, and clear (1–2 lines). Avoid legal jargon.

--- RELEVANT CLAUSES ---
{chr(10).join(top_clauses)}

--- QUESTION ---
{question}

--- ANSWER ---"""

            try:
                response = gemini_model.generate_content(prompt)
                answers.append(response.text.strip())
            except Exception as e:
                answers.append(f"Error: {str(e)}")

        return jsonify({
            "answers": answers,
            "response_time_seconds": round(time.time() - start_time, 2)
        })

    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))




