from fastapi.testclient import TestClient
from FrontendMS.app import app
import pytest

client = TestClient(app)


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_chat_page():
    response = client.get("/chat")
    assert response.status_code == 200
    assert "DeepWriter" in response.text


