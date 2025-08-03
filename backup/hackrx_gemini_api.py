import os
from flask import Flask, request, jsonify
from hackrx_qa_engine import get_answers  # Make sure this function works as expected

app = Flask(__name__)

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    # 🔐 Auth check
    auth_header = request.headers.get("Authorization")
    expected_token = f"Bearer {os.getenv('hackrx')}"
    if auth_header != expected_token:
        return jsonify({"detail": "Forbidden"}), 403

    try:
        data = request.get_json()
        documents_url = data["documents"]
        questions = data["questions"]

        answers = get_answers(documents_url, questions)
        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
