"""Tests for the FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
def app():
    """Create a test app."""
    return create_app()


@pytest.fixture
async def client(app):
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestRootEndpoint:
    """Test the root endpoint."""

    @pytest.mark.asyncio
    async def test_root(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Flux-AI"
        assert data["version"] == "0.1.0"


class TestHealthEndpoint:
    """Test the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_status(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "services" in data


class TestChatEndpoint:
    """Test the chat endpoints."""

    @pytest.mark.asyncio
    async def test_chat_sync_requires_body(self, client):
        resp = await client.post("/chat/sync")
        assert resp.status_code == 422  # Validation error — no body


class TestDocumentsEndpoint:
    """Test the documents endpoints."""

    @pytest.mark.asyncio
    async def test_list_documents(self, client):
        resp = await client.get("/documents/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
