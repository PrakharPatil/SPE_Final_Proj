from fastapi.testclient import TestClient
from TransformerMS.main import app
import pytest

client = TestClient(app)

@pytest.mark.parametrize("prompt,expected_status", [
    ("What is AI?", 200),
    ("", 422),  # Should fail validation
    ("P" * 1001, 422),  # Exceeds max length
    ("Hello @world!", 200)
])
def test_generate_endpoint(prompt, expected_status):
    response = client.post(
        "/generate",
        json={"prompt": prompt}
    )
    assert response.status_code == expected_status
    if expected_status == 200:
        assert "generated_text" in response.json()
        assert len(response.json()["generated_text"]) > 0


