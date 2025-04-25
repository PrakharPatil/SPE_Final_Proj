# Example request using Python requests
import requests

response = requests.post(
    "http://localhost:8081/generate",
    json={"prompt": "Machine Learning "}
)

print(response.json())
# {"generated_text": "The meaning of life is to understand the fundamental nature of reality..."}