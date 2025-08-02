from flask import Flask, request, jsonify
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import time  # <--- added

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

@app.route('/hackrx/run', methods=['POST'])
def run():
    try:
        start_time = time.time()  # <--- start timing

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        data = request.get_json()
        pdf_url = data.get("documents")
        questions = data.get("questions", [])

        if not pdf_url or not questions:
            return jsonify({'error': 'Missing PDF URL or questions'}), 400

        # Download the PDF
        pdf_response = requests.get(pdf_url, timeout=15)
        if pdf_response.status_code != 200:
            return jsonify({'error': f'Failed to download PDF from URL: {pdf_url}'}), 400

        with open("temp.pdf", "wb") as f:
            f.write(pdf_response.content)

        # Extract text from PDF
        doc = fitz.open("temp.pdf")
        full_text = ""
        for i in range(len(doc)):
            try:
                page = doc.load_page(i)
                full_text += page.get_text()
            except Exception as e:
                print(f"Warning: Skipped page {i} due to error: {e}")
        doc.close()

        if not full_text.strip():
            return jsonify({'error': 'No text extracted from PDF'}), 400

        # Generate answers
        answers = []
        for question in questions:
            prompt = f"""
You are a smart insurance assistant. Based only on the insurance policy document below, give a direct, short answer (1–2 lines) to the question, written in simple, understandable language. Avoid legal wording and long definitions.

---DOCUMENT---
{full_text}

---QUESTION---
{question}

---ANSWER---"""
            try:
                result = gemini_model.generate_content(prompt)
                answer = result.text.strip()
            except Exception as e:
                answer = f"Error: {str(e)}"
            answers.append(answer)

        total_time = round(time.time() - start_time, 2)  # <--- end timing

        return jsonify({
            "answers": answers,
            "response_time_seconds": total_time  # <--- include response time
        })

    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

