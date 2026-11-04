"""Unit tests for the PhishGuard FastAPI backend."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


# ── Fixture ─────────────────────────────────────────────────────

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── 1. Root endpoint ───────────────────────────────────────────

class TestRootEndpoint:
    @pytest.mark.anyio
    async def test_status_code(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_project_name(self, client):
        data = (await client.get("/")).json()
        assert data["project"] == "PhishGuard"

    @pytest.mark.anyio
    async def test_status_online(self, client):
        data = (await client.get("/")).json()
        assert data["status"] == "online"

    @pytest.mark.anyio
    async def test_version_present(self, client):
        data = (await client.get("/")).json()
        assert "version" in data

    @pytest.mark.anyio
    async def test_response_schema_keys(self, client):
        data = (await client.get("/")).json()
        assert set(data.keys()) == {"project", "status", "version"}


# ── 2. Health endpoint ─────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.anyio
    async def test_status_code(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_healthy(self, client):
        data = (await client.get("/health")).json()
        assert data["status"] == "healthy"

    @pytest.mark.anyio
    async def test_service_name(self, client):
        data = (await client.get("/health")).json()
        assert data["service"] == "PhishGuard API"

    @pytest.mark.anyio
    async def test_response_schema_keys(self, client):
        data = (await client.get("/health")).json()
        assert set(data.keys()) == {"status", "service"}


# ── 3. Error handling ──────────────────────────────────────────

class TestErrorHandling:
    @pytest.mark.anyio
    async def test_404_on_unknown_route(self, client):
        resp = await client.get("/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_404_body_has_detail(self, client):
        data = (await client.get("/nonexistent")).json()
        assert "detail" in data

    @pytest.mark.anyio
    async def test_method_not_allowed(self, client):
        resp = await client.post("/")
        assert resp.status_code == 405


# ── 4. CORS headers ────────────────────────────────────────────

class TestCORS:
    @pytest.mark.anyio
    async def test_cors_allows_localhost(self, client):
        resp = await client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    @pytest.mark.anyio
    async def test_cors_allows_127(self, client):
        resp = await client.options(
            "/health",
            headers={
                "Origin": "http://127.0.0.1:8000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    @pytest.mark.anyio
    async def test_cors_allows_chrome_extension(self, client):
        resp = await client.options(
            "/analyze",
            headers={
                "Origin": "chrome-extension://knldjmfmopnppmplflldnbnekgfdnmfi",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "chrome-extension://knldjmfmopnppmplflldnbnekgfdnmfi"


# ── 5. Content type ────────────────────────────────────────────

class TestContentType:
    @pytest.mark.anyio
    async def test_root_returns_json(self, client):
        resp = await client.get("/")
        assert "application/json" in resp.headers["content-type"]

    @pytest.mark.anyio
    async def test_health_returns_json(self, client):
        resp = await client.get("/health")
        assert "application/json" in resp.headers["content-type"]

