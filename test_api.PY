import requests

headers = {
    "Authorization": "Bearer hackrx123",
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

res = requests.post("https://test1-mqan.onrender.com/hackrx/run", json=data, headers=headers)
print("Status:", res.status_code)
print("Response:", res.json())

