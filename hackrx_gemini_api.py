from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import requests
import google.generativeai as genai
import numpy as np
import tempfile
import os

# Initialize Flask app
app = Flask(__name__)

# ✅ Directly set Gemini API Key (NOT RECOMMENDED for production)
GEMINI_API_KEY = "AIzaSyCOxSvEYOT3eQaCT21SFwcK-3klYPf_KnI"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunking utility
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Extract text from PDF URL
def extract_text_from_pdf_url(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

# Create FAISS index
def create_faiss_index(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

# Retrieve top matching chunks
def get_top_k_chunks(query, chunks, embeddings, index, k=1):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

# Ask Gemini with relevant clause
def ask_gemini(question, context):
    prompt = f"""
Answer the following insurance-related question using ONLY the context provided. Respond in 1–2 clear sentences. If not found, say "Not mentioned".

Context:
\"\"\"
{context}
\"\"\"

Question:
{question}

Answer:"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# Main HackRx endpoint
@app.route('/hackrx/run', methods=['POST'])
def run_hackrx():
    data = request.get_json()
    pdf_url = data.get('pdf_url') or data.get('documents')
    questions = data.get('questions')

    if not pdf_url or not questions:
        return jsonify({"error": "Missing 'pdf_url' or 'questions'"}), 400

    full_text = extract_text_from_pdf_url(pdf_url)
    if not full_text:
        return jsonify({"error": "Unable to extract text from the document"}), 400

    chunks = chunk_text(full_text)
    index, embeddings = create_faiss_index(chunks)

    results = []
    for question in questions:
        top_chunks = get_top_k_chunks(question, chunks, embeddings, index, k=1)
        matched_chunk = top_chunks[0] if top_chunks else ""
        answer = ask_gemini(question, matched_chunk)
        results.append({
            "question": question,
            "answer": answer,
            "matched_clause": matched_chunk
        })

    return jsonify({"answers": results})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))



