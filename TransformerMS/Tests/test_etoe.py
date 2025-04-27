import pytest
from httpx import AsyncClient

# Import the FastAPI app and model loader from the backend service.
# (Adjust the import paths to match your project structure.)
from TransformerMS.main import app as backend_app, model_loader


@pytest.mark.asyncio
async def test_generate_endpoint_unit(monkeypatch):
    """
    Unit test: Mock the model generation and verify /generate returns the mocked text.
    """
    # Mock the heavy generate_response method
    monkeypatch.setattr(model_loader, "generate_response", lambda prompt: "mocked output")

    async with AsyncClient(app=backend_app, base_url="http://test") as client:
        response = await client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == 200
    data = response.json()
    # Check that 'generated_text' field is present and equals the mocked output
    assert "generated_text" in data
    assert data["generated_text"] == "mocked output"


@pytest.mark.asyncio
async def test_generate_endpoint_integration(monkeypatch):
    """
    Integration test: Call the actual endpoint and check response structure.
    The model_loader.generate_response is still patched to avoid heavy computation.
    """
    # Use a dummy implementation to simulate the model
    monkeypatch.setattr(model_loader, "generate_response", lambda prompt: f"echo: {prompt}")

    async with AsyncClient(app=backend_app, base_url="http://test") as client:
        response = await client.post("/generate", json={"prompt": "Test prompt"})

    assert response.status_code == 200
    data = response.json()
    # Validate response contains the generated_text field
    assert "generated_text" in data
    # The content should reflect the prompt as per our dummy implementation
    assert data["generated_text"] == "echo: Test prompt"
