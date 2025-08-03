services:
  - type: web
    name: hackrx-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python hackrx_gemini.py"
    envVars:
      - key: PIKACHU
        fromEnv: PIKACHU              # Gemini API Key
      - key: hackrx
        fromEnv: hackrx               # API authorization key
