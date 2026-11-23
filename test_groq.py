import tomllib
from pathlib import Path
from groq import Groq

# Locate secrets file relative to this script
secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"

if not secrets_path.exists():
    raise FileNotFoundError(f"Secrets file not found at: {secrets_path}")

with open(secrets_path, "rb") as f:
    secrets = tomllib.load(f)

api_key = secrets.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not defined in secrets.toml")

client = Groq(api_key=api_key)

print("Connecting to Groq LLM (llama-3.3-70b-versatile)...")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Explain sentiment analysis in NLP in 3 simple sentences."
        }
    ],
    temperature=0.3,
    max_tokens=200
)

print("\n========== LLM RESPONSE ==========\n")
print(response.choices[0].message.content)
print("\n========== TEST SUCCESSFUL ==========")
