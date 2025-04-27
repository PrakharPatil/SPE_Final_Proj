import pytest
from httpx import AsyncClient

# Import the FastAPI app from the frontend service.
# (Adjust the import path to your frontend application module.)
from FrontendMS.app import app as frontend_app


# We also import or define a dummy httpx.AsyncClient to mock backend calls.
class DummyResponse:
    def __init__(self, json_data):
        self._json = json_data
        self.status_code = 200

    def json(self):
        return self._json


class DummyAsyncClient:
    """
    Dummy AsyncClient to mock httpx.AsyncClient for backend calls.
    """

    async def post(self, url, json):
        # Return a dummy response object with a JSON method
        return DummyResponse({"generated_text": "dummy from backend"})

    async def __aenter__(self): return self

    async def __aexit__(self, exc_type, exc, tb): pass


@pytest.mark.asyncio
async def test_html_endpoints_render(monkeypatch):
    """
    Unit test: Verify that '/' and '/chat' endpoints return status 200 and HTML.
    """
    async with AsyncClient(app=frontend_app, base_url="http://test") as client:
        res_index = await client.get("/")
        res_chat = await client.get("/chat")
    # Both should be successful and return HTML content
    assert res_index.status_code == 200
    assert "text/html" in res_index.headers.get("content-type", "")
    assert res_chat.status_code == 200
    assert "text/html" in res_chat.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_api_generate_unit(monkeypatch):
    """
    Unit test: Mock backend call for /api/generate and verify the forwarded response.
    """
    # Monkey-patch httpx.AsyncClient used in the endpoint to our dummy client
    monkeypatch.setattr("FrontendMS.app.httpx.AsyncClient", lambda: DummyAsyncClient())

    async with AsyncClient(app=frontend_app, base_url="http://test") as client:
        response = await client.post("/api/generate", json={"prompt": "Hello Frontend"})

    assert response.status_code == 200
    data = response.json()
    # The frontend should forward the dummy backend response exactly
    assert data == {"generated_text": "dummy from backend"}


@pytest.mark.asyncio
async def test_api_generate_integration(monkeypatch):
    """
    Integration test: Simulate user sending a prompt to /api/generate.
    Backend call is mocked so that we test the end-to-end path.
    """
    # Use the same dummy backend client as above
    monkeypatch.setattr("FrontendMS.app.httpx.AsyncClient", lambda: DummyAsyncClient())

    async with AsyncClient(app=frontend_app, base_url="http://test") as client:
        response = await client.post("/api/generate", json={"prompt": "Another test"})

    assert response.status_code == 200
    data = response.json()
    # Confirm the structure and content of the response
    assert "generated_text" in data
    assert data["generated_text"] == "dummy from backend"
