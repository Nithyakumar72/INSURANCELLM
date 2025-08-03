from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import requests
import google.generativeai as genai
import numpy as np
import tempfile
import os

app = Flask(__name__)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # <-- Replace with actual key
gemini_model = genai.GenerativeModel("gemini-pro")

# Utility: Split document into word chunks
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Extract text from PDF (given URL)
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
        return None

# Create FAISS index of embedded chunks
def create_faiss_index(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

# Retrieve top K relevant chunks from index
def get_top_k_chunks(query, chunks, embeddings, index, k=1):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

# Ask Gemini using the top matched clause
def ask_gemini(question, context):
    prompt = f"""
Answer the following insurance-related question based ONLY on the provided context below. Respond in 1–2 short sentences. If not found, reply 'Not mentioned'.

Context:
\"\"\"
{context}
\"\"\"

Question:
{question}

Answer:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "Error: " + str(e)

# Main HackRx endpoint
@app.route('/hackrx/run', methods=['POST'])
def run_hackrx():
    data = request.get_json()
    pdf_url = data.get('pdf_url') or data.get('documents')  # Accept both keys
    questions = data.get('questions')

    if not pdf_url or not questions:
        return jsonify({"error": "Missing pdf_url/documents or questions"}), 400

    # Step 1: Extract and chunk PDF text
    full_text = extract_text_from_pdf_url(pdf_url)
    if not full_text:
        return jsonify({"error": "Failed to extract PDF text"}), 400

    chunks = chunk_text(full_text)
    index, embeddings = create_faiss_index(chunks)

    # Step 2: Answer questions
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


