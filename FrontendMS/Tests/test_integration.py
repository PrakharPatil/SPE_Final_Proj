# tests/frontend/integration/test_integration.py
import pytest
from fastapi.testclient import TestClient
from fastapi import status
from httpx import Response
from FrontendMS.app import app  # Your frontend FastAPI app
import respx

client = TestClient(app)


@pytest.fixture
def mock_backend():
    with respx.mock(base_url="http://localhost:8081") as m:
        yield m


@pytest.mark.parametrize("prompt,expected_response", [
    ("What is AI?", {"generated_text": "AI stands for Artificial Intelligence..."}),
    ("Explain machine learning", {"generated_text": "Machine learning is a subset..."}),
])
def test_valid_prompt_proxy(mock_backend, prompt, expected_response):
    # Mock the backend response correctly
    mock_backend.post("/generate").mock(
        return_value=Response(
            status_code=status.HTTP_200_OK,
            json=expected_response
        )
    )

    response = client.post(
        "/api/generate",
        json={"prompt": prompt}
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


def test_empty_prompt_validation():
    response = client.post(
        "/api/generate",
        json={"prompt": ""}
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert any(
        error["loc"] == ["body", "prompt"] and
        "ensure this value has at least 1 character" in error["msg"]
        for error in response.json()["detail"]
    )


def test_long_prompt_validation():
    long_prompt = "A" * 1001  # Assuming max_length=1000 in your model
    response = client.post(
        "/api/generate",
        json={"prompt": long_prompt}
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert any(
        error["loc"] == ["body", "prompt"] and
        "ensure this value has at most 1000 characters" in error["msg"]
        for error in response.json()["detail"]
    )


def test_backend_error_handling(mock_backend):
    mock_backend.post("/generate").mock(
        return_value=Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            json={"detail": "Internal server error"}
        )
    )

    response = client.post(
        "/api/generate",
        json={"prompt": "test"}
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.json() == {"detail": "Internal server error"}


def test_rate_limited_requests(mock_backend):
    mock_backend.post("/generate").mock(
        return_value=Response(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            json={"detail": "Too many requests"}
        )
    )

    response = client.post(
        "/api/generate",
        json={"prompt": "test"}
    )

    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Too many requests" in response.json()["detail"]


def test_home_page_rendering():
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response.headers["content-type"]
    assert "DeepWriter" in response.text


def test_chat_page_rendering():
    response = client.get("/chat")
    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response.headers["content-type"]
    assert "chat-container" in response.text
    assert "message-box" in response.text