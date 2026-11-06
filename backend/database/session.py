"""
PhishGuard – Database Session & Engine Configuration
"""

import logging
import os
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.database.models import Base

logger = logging.getLogger(__name__)

# Project root: fishGuard/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "phishguard.db")

# Required columns for the 'analysis' table schema
ANALYSIS_MIGRATION_COLUMNS: dict[str, str] = {
    "threat_category": "VARCHAR",
    "class_probabilities": "JSON",
    "top_features": "JSON",
    "explanation": "JSON",
    "visual_status": "VARCHAR DEFAULT 'not_provided'",
    "visual_features": "JSON",
}


class DatabaseMigrationError(RuntimeError):
    """Raised when an automated schema migration or view creation fails."""


def get_db_path() -> str:
    """Return the active database path from environment or default."""
    return os.getenv("PHISHGUARD_DB_PATH", DEFAULT_DB_PATH)


def build_db_url(db_path: str | None = None) -> str:
    """Construct SQLite connection URL."""
    path = db_path if db_path is not None else get_db_path()
    if path == ":memory:" or path.startswith("sqlite:"):
        return path if path.startswith("sqlite:") else f"sqlite:///{path}"
    # Normalize absolute path with forward slashes for SQLite URL
    normalized = os.path.abspath(path).replace("\\", "/")
    return f"sqlite:///{normalized}"


def create_db_engine(db_path: str | None = None) -> Engine:
    """Create a new SQLAlchemy engine for the specified database path."""
    url = build_db_url(db_path)
    # Ensure directory exists if using file-based SQLite
    path = db_path if db_path is not None else get_db_path()
    if path != ":memory:" and not path.startswith("sqlite:"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    return create_engine(
        url,
        connect_args={"check_same_thread": False},
    )


# Default engine & sessionmaker
engine: Engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def migrate_db(target_engine: Engine) -> list[str]:
    """Inspect the 'analysis' table and apply missing column migrations.

    Returns the list of column names successfully added.
    Raises DatabaseMigrationError if schema inspection or column addition fails.
    """
    applied_columns: list[str] = []
    try:
        from sqlalchemy import inspect
        inspector = inspect(target_engine)
        if "analysis" not in inspector.get_table_names():
            logger.debug("Table 'analysis' does not exist yet; skipping column migration.")
            return applied_columns

        existing_columns = {c["name"] for c in inspector.get_columns("analysis")}
    except SQLAlchemyError as exc:
        logger.error("Failed to inspect database schema before migration: %s", exc)
        raise DatabaseMigrationError(f"Database schema inspection failed: {exc}") from exc

    for col_name, col_type in ANALYSIS_MIGRATION_COLUMNS.items():
        if col_name not in existing_columns:
            logger.info("Migrating table 'analysis': adding missing column '%s' (%s)", col_name, col_type)
            try:
                with target_engine.connect() as conn:
                    conn.execute(text(f"ALTER TABLE analysis ADD COLUMN {col_name} {col_type};"))
                    conn.commit()
                applied_columns.append(col_name)
            except SQLAlchemyError as exc:
                logger.error(
                    "Database migration failed while adding column '%s' (%s) to 'analysis': %s",
                    col_name,
                    col_type,
                    exc,
                )
                raise DatabaseMigrationError(
                    f"Failed to add column '{col_name}' ({col_type}) to 'analysis': {exc}"
                ) from exc

    return applied_columns


def refresh_views(target_engine: Engine) -> None:
    """Create or refresh compatibility view 'analyses' -> 'analysis'."""
    try:
        with target_engine.connect() as conn:
            conn.execute(text("DROP VIEW IF EXISTS analyses;"))
            conn.execute(text("CREATE VIEW analyses AS SELECT * FROM analysis;"))
            conn.commit()
    except SQLAlchemyError as exc:
        logger.error("Failed to create or refresh compatibility view 'analyses': %s", exc)
        raise DatabaseMigrationError(
            f"Failed to create compatibility view 'analyses': {exc}"
        ) from exc


def init_db(
    db_path: str | None = None,
    engine_override: Engine | None = None,
) -> Engine:
    """Initialize database tables, apply backward-compatible column migrations, and recreate view."""
    if engine_override:
        target_engine = engine_override
    elif db_path:
        target_engine = create_db_engine(db_path)
    else:
        current_path = get_db_path()
        target_engine = create_db_engine(current_path)

    try:
        Base.metadata.create_all(bind=target_engine)
    except SQLAlchemyError as exc:
        logger.error("Failed to create database tables: %s", exc)
        raise DatabaseMigrationError(f"Failed to create database tables: {exc}") from exc

    migrate_db(target_engine)
    refresh_views(target_engine)

    return target_engine


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions."""
    current_engine = create_db_engine(get_db_path())
    CurrentSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=current_engine)
    db = CurrentSessionLocal()
    try:
        yield db
    finally:
        db.close()
