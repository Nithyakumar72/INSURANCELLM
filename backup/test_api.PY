import requests

API_URL = "https://hackrx-api.onrender.com/hackrx/run"  # Replace with your actual deployed Render URL if different
API_KEY = "d464d88731074c5923019b6139916b4ba2e7cd1b8cb01316fed78295b75c066e"  # Your HACKRX_API_KEY

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "documents": "https://raw.githubusercontent.com/WAYNE-INNOVATION-23/hackx/main/BAJHLIP23020V012223.pdf",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

try:
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    print("✅ Status:", response.status_code)
    print("🧠 Answers:")
    for i, answer in enumerate(response.json().get("answers", []), start=1):
        print(f"Q{i}: {data['questions'][i-1]}")
        print(f"A{i}: {answer}\n")
except requests.exceptions.RequestException as e:
    print("❌ Request failed:", e)
    if 'response' in locals():
        print("Raw response:", response.text)
