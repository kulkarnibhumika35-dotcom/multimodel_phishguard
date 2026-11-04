"""
PhishGuard – FastAPI Application

Start with:
    python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import analyze_router, base_router
from backend.database.session import init_db
from backend.services.analyzer import init_analyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Automatically create database tables and safely load ML model and explainer at startup."""
    init_db()
    init_analyzer()
    yield


# ── Application ─────────────────────────────────────────────────

app = FastAPI(
    title="PhishGuard API",
    description="AI-powered phishing URL detection service.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS – allow Chrome extensions & localhost dev ──────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8501",     # Streamlit dashboard
        "http://127.0.0.1:8501",
    ],
    allow_origin_regex=r"^chrome-extension://[a-z0-9]+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────────────────

app.include_router(base_router)
app.include_router(analyze_router, prefix="")

# ── Global error handler ───────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all so unhandled errors return structured JSON."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc),
        },
    )
