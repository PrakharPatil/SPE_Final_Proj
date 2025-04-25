# Example request using Python requests
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "The meaning of life is"}
)

print(response.json())
# {"generated_text": "The meaning of life is to understand the fundamental nature of reality..."}