# hackrx_gemini_api.py

import os
import fitz  # PyMuPDF
import json
from flask import Flask, request, jsonify
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# Setup Gemini with your actual API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Set this in Render's environment variables
model = genai.GenerativeModel("models/gemini-1.5-flash")

app = Flask(__name__)
PDF_PATH = "BAJHLIP23020V012223.pdf"  # Place this in your Render project root

# Step 1: Load PDF and Build FAISS Index
def load_and_index_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.from_texts(chunks, embeddings)
    return db

db = load_and_index_pdf(PDF_PATH)

# Step 2: Define /hackrx/run Endpoint
@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    if request.headers.get("Authorization") != f"Bearer {os.getenv('HACKRX_API_KEY')}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    questions = data.get("questions", [])
    answers = []

    for question in questions:
        docs = db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are a healthcare policy expert helping users understand their insurance documents.

Context:
{context}

Question:
{question}

Instructions:
- Provide a clear, concise answer.
- If the information is not available in the context, say "Not specified in the document."
- Avoid assumptions or generalizations.
- Keep the answer within 1-2 sentences.
- Respond only with the answer to the question.

Answer:"""

        
        try:
            response = model.generate_content(prompt)
            answers.append(response.text.strip())
        except Exception as e:
            answers.append(f"Error: {str(e)}")

    return jsonify({"answers": answers})

# Run locally (for debugging only)
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
