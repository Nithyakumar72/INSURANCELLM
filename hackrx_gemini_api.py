import os
import json
import google.generativeai as genai
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY", "hackrx123")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Load documents and generate embeddings
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            documents.append({"filename": filename, "content": text})
    return documents

documents = load_documents("data/")
clauses = [{"filename": doc["filename"], "clause": c} for doc in documents for c in doc["content"].split("\n") if len(c.strip()) > 50]

# Generate embeddings
model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)
clause_texts = [c["clause"] for c in clauses]
clause_embeddings = embedder.encode(clause_texts, convert_to_tensor=True)

@app.post("/ask/")
async def ask_question(request: Request, query: QueryRequest):
    if request.headers.get("X-API-KEY") != HACKRX_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

    query_embedding = embedder.encode([query.query], convert_to_tensor=True)
    scores = cosine_similarity(query_embedding, clause_embeddings)[0]
    top_indices = scores.argsort()[-3:][::-1]

    top_clauses = [clauses[i] for i in top_indices]
    context = "\n\n".join([f"{c['clause']}" for c in top_clauses])

    prompt = f"Based on the following document clauses, answer the question:\n\n{context}\n\nQuestion: {query.query}"
    response = model.generate_content(prompt)

    return {
        "query": query.query,
        "matched_clauses": top_clauses,
        "answer": response.text
    }





