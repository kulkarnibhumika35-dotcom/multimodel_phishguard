"""Unit tests for PhishGuard SQLite database integration."""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from backend.database.crud import (
    create_analysis,
    get_analysis,
    get_analysis_by_id,
    list_analyses,
    list_records,
)
from backend.database.models import Analysis, Base
from backend.database.session import (
    ANALYSIS_MIGRATION_COLUMNS,
    DatabaseMigrationError,
    build_db_url,
    create_db_engine,
    get_db,
    get_db_path,
    init_db,
    migrate_db,
    refresh_views,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path):
    """Create an isolated SQLite database for each test in a temporary folder."""
    db_file = str(tmp_path / "test_phishguard.db")
    engine = create_db_engine(db_file)
    init_db(engine_override=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()

    try:
        yield session, engine, db_file
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def db_session(temp_db):
    """Convenience fixture returning just the active session."""
    session, _, _ = temp_db
    return session


# ── 1. Model & Schema Verification ─────────────────────────────

class TestDatabaseSchema:
    def test_tables_created(self, temp_db):
        _, engine, _ = temp_db
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "analysis" in tables

    def test_table_columns(self, temp_db):
        _, engine, _ = temp_db
        inspector = inspect(engine)
        columns = {col["name"]: col for col in inspector.get_columns("analysis")}

        expected_columns = {
            "id",
            "url",
            "prediction",
            "risk_score",
            "analysis_status",
            "analyzed_at",
        }
        assert expected_columns.issubset(columns.keys())

    def test_primary_key(self, temp_db):
        _, engine, _ = temp_db
        inspector = inspect(engine)
        pk = inspector.get_pk_constraint("analysis")
        assert pk["constrained_columns"] == ["id"]


# ── 2. CRUD: Create Analysis Record ────────────────────────────

class TestCreateAnalysis:
    def test_create_minimal_record(self, db_session):
        record = create_analysis(db_session, url="https://example.com")
        assert record.id is not None
        assert record.url == "https://example.com"
        assert record.prediction is None
        assert record.risk_score is None
        assert record.analysis_status == "completed"
        assert isinstance(record.analyzed_at, datetime)

    def test_create_phishing_record(self, db_session):
        record = create_analysis(
            db_session,
            url="http://malicious-login.xyz/verify",
            prediction=1,
            risk_score=0.97,
            analysis_status="completed",
        )
        assert record.id is not None
        assert record.prediction == 1
        assert record.risk_score == 0.97
        assert record.analysis_status == "completed"

    def test_create_legitimate_record(self, db_session):
        record = create_analysis(
            db_session,
            url="https://google.com",
            prediction=0,
            risk_score=0.03,
            analysis_status="completed",
        )
        assert record.id is not None
        assert record.prediction == 0
        assert record.risk_score == 0.03

    def test_create_malware_record_with_threat_category(self, db_session):
        record = create_analysis(
            db_session,
            url="http://192.168.1.1/payload.exe",
            prediction=2,
            threat_category="malware_distribution",
            risk_score=0.99,
            analysis_status="completed",
        )
        assert record.id is not None
        assert record.prediction == 2
        assert record.threat_category == "malware_distribution"
        assert record.risk_score == 0.99

    def test_create_with_custom_timestamp(self, db_session):
        custom_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        record = create_analysis(
            db_session,
            url="https://example.org",
            analyzed_at=custom_time,
        )
        # SQLite stores naive datetimes; verify year/month/day/hour match
        assert record.analyzed_at.year == 2025
        assert record.analyzed_at.month == 1
        assert record.analyzed_at.day == 1


# ── 3. CRUD: Get Analysis Record ───────────────────────────────

class TestGetAnalysis:
    def test_get_by_id_exists(self, db_session):
        created = create_analysis(
            db_session,
            url="https://test.com",
            prediction=0,
            risk_score=0.15,
        )
        retrieved = get_analysis_by_id(db_session, created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.url == "https://test.com"
        assert retrieved.prediction == 0
        assert retrieved.risk_score == 0.15

    def test_get_by_id_nonexistent(self, db_session):
        result = get_analysis_by_id(db_session, 99999)
        assert result is None

    def test_get_analysis_alias(self, db_session):
        created = create_analysis(db_session, url="https://alias.test")
        retrieved = get_analysis(db_session, created.id)
        assert retrieved is not None
        assert retrieved.id == created.id


# ── 4. CRUD: List Analyses (Newest First) ──────────────────────

class TestListAnalyses:
    def test_list_empty(self, db_session):
        records = list_analyses(db_session)
        assert records == []

    def test_list_ordered_newest_first(self, db_session):
        base_time = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        r1 = create_analysis(
            db_session,
            url="https://first.com",
            analyzed_at=base_time,
        )
        r2 = create_analysis(
            db_session,
            url="https://second.com",
            analyzed_at=base_time + timedelta(hours=1),
        )
        r3 = create_analysis(
            db_session,
            url="https://third.com",
            analyzed_at=base_time + timedelta(hours=2),
        )

        records = list_analyses(db_session)
        assert len(records) == 3
        assert records[0].id == r3.id
        assert records[1].id == r2.id
        assert records[2].id == r1.id

    def test_list_pagination_limit(self, db_session):
        for i in range(5):
            create_analysis(db_session, url=f"https://site{i}.com")

        records = list_analyses(db_session, limit=2)
        assert len(records) == 2

    def test_list_pagination_skip(self, db_session):
        for i in range(5):
            create_analysis(db_session, url=f"https://site{i}.com")

        all_records = list_analyses(db_session, limit=10)
        skipped_records = list_analyses(db_session, skip=2, limit=10)
        assert len(skipped_records) == 3
        assert skipped_records[0].id == all_records[2].id

    def test_list_records_alias(self, db_session):
        create_analysis(db_session, url="https://alias.test")
        assert len(list_records(db_session)) == 1


# ── 5. Configurable Database Path ──────────────────────────────

class TestConfigurableDatabasePath:
    def test_custom_db_creation(self, tmp_path):
        custom_file = str(tmp_path / "custom" / "my_custom.db")
        engine = create_db_engine(custom_file)
        init_db(engine_override=engine)

        assert os.path.isfile(custom_file)
        inspector = inspect(engine)
        assert "analysis" in inspector.get_table_names()
        engine.dispose()

    def test_env_var_override(self, monkeypatch, tmp_path):
        env_file = str(tmp_path / "env_override.db")
        monkeypatch.setenv("PHISHGUARD_DB_PATH", env_file)
        assert get_db_path() == env_file

        url = build_db_url()
        assert "env_override.db" in url

    def test_get_db_generator(self):
        gen = get_db()
        session = next(gen)
        assert session is not None
        # Clean up generator
        try:
            next(gen)
        except StopIteration:
            pass


# ── 6. FastAPI Lifespan Table Initialization ───────────────────

class TestFastAPITablesInit:
    @pytest.mark.anyio
    async def test_lifespan_creates_tables(self, tmp_path, monkeypatch):
        test_db_file = str(tmp_path / "lifespan_test.db")
        monkeypatch.setenv("PHISHGUARD_DB_PATH", test_db_file)

        # Import lifespan and trigger it
        from backend.main import lifespan, app

        async with lifespan(app):
            # Tables must be created during lifespan startup
            assert os.path.isfile(test_db_file)
            engine = create_db_engine(test_db_file)
            inspector = inspect(engine)
            assert "analysis" in inspector.get_table_names()
            engine.dispose()


# ── 7. SQLite Migration & Schema Compatibility ──────────────────

class TestDatabaseMigrations:
    def test_fresh_database_initialization(self, tmp_path):
        """Requirement 6 & 7: Verify fresh DB initializes all columns and views."""
        db_path = str(tmp_path / "fresh_phishguard.db")
        engine = init_db(db_path=db_path)

        inspector = inspect(engine)
        assert "analysis" in inspector.get_table_names()
        assert "analyses" in inspector.get_view_names()

        # Check all required columns exist in analysis table
        table_cols = {c["name"] for c in inspector.get_columns("analysis")}
        expected_cols = {
            "id",
            "url",
            "prediction",
            "threat_category",
            "risk_score",
            "class_probabilities",
            "top_features",
            "explanation",
            "analysis_status",
            "analyzed_at",
        }
        assert expected_cols.issubset(table_cols)

        # Check all required columns exist in analyses view
        view_cols = {c["name"] for c in inspector.get_columns("analyses")}
        assert expected_cols.issubset(view_cols)

        engine.dispose()

    def test_legacy_database_migration_and_data_preservation(self, tmp_path):
        """Requirements 4 & 5: Verify migration against an actual legacy SQLite DB
        containing only original columns, ensuring existing records remain intact.
        """
        import sqlite3

        legacy_file = str(tmp_path / "legacy_v1.db")
        con = sqlite3.connect(legacy_file)
        cur = con.cursor()

        # 1. Create the legacy 6-column table schema
        cur.execute(
            """
            CREATE TABLE analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url VARCHAR NOT NULL,
                prediction INTEGER,
                risk_score FLOAT,
                analysis_status VARCHAR NOT NULL,
                analyzed_at DATETIME NOT NULL
            );
        """
        )

        # 2. Insert pre-existing records into the legacy schema
        cur.execute(
            """
            INSERT INTO analysis (url, prediction, risk_score, analysis_status, analyzed_at)
            VALUES ('https://legacy-legit.com', 0, 0.05, 'completed', '2026-11-01 10:00:00');
        """
        )
        cur.execute(
            """
            INSERT INTO analysis (url, prediction, risk_score, analysis_status, analyzed_at)
            VALUES ('https://legacy-phish.com/login', 1, 0.95, 'completed', '2026-11-01 11:00:00');
        """
        )
        cur.execute(
            """
            INSERT INTO analysis (url, prediction, risk_score, analysis_status, analyzed_at)
            VALUES ('https://legacy-unavail.com', NULL, NULL, 'model_unavailable', '2026-11-01 12:00:00');
        """
        )
        con.commit()

        # Verify old columns prior to migration
        initial_cols = [c[1] for c in cur.execute("PRAGMA table_info(analysis)").fetchall()]
        assert "threat_category" not in initial_cols
        assert "class_probabilities" not in initial_cols
        assert "top_features" not in initial_cols
        assert "explanation" not in initial_cols
        con.close()

        # 3. Apply migration via init_db
        engine = init_db(db_path=legacy_file)

        # 4. Verify all new columns exist post-migration
        inspector = inspect(engine)
        post_cols = {c["name"] for c in inspector.get_columns("analysis")}
        for new_col in ANALYSIS_MIGRATION_COLUMNS:
            assert new_col in post_cols

        # 5. Verify existing records remain completely intact (Requirement 5)
        TestingSession = sessionmaker(bind=engine)
        db = TestingSession()
        try:
            records = list_analyses(db)
            assert len(records) == 3

            r_legit = db.query(Analysis).filter(Analysis.url == "https://legacy-legit.com").one()
            assert r_legit.prediction == 0
            assert r_legit.risk_score == 0.05
            assert r_legit.analysis_status == "completed"

            r_phish = db.query(Analysis).filter(Analysis.url == "https://legacy-phish.com/login").one()
            assert r_phish.prediction == 1
            assert r_phish.risk_score == 0.95
            assert r_phish.analysis_status == "completed"

            r_unavail = db.query(Analysis).filter(Analysis.url == "https://legacy-unavail.com").one()
            assert r_unavail.prediction is None
            assert r_unavail.risk_score is None
            assert r_unavail.analysis_status == "model_unavailable"

            # 6. Verify we can insert new records with new fields into the migrated table
            new_record = create_analysis(
                db=db,
                url="http://192.168.1.1:8080/setup.exe",
                prediction=2,
                threat_category="malware_distribution",
                risk_score=0.99,
                class_probabilities={"legitimate": 0.0, "phishing": 0.0, "malware_distribution": 1.0},
                top_features=[{"feature_name": "num_digits", "contribution": 0.2}],
                explanation={"status": "explained"},
            )
            assert new_record.id is not None
            assert new_record.threat_category == "malware_distribution"
            assert new_record.class_probabilities["malware_distribution"] == 1.0
        finally:
            db.close()
            engine.dispose()

    def test_migration_failure_raises_descriptive_error(self, tmp_path, monkeypatch):
        """Requirement 8: Ensure schema migration failure raises DatabaseMigrationError and does not silently pass."""
        import sqlite3
        from sqlalchemy.exc import OperationalError

        db_path = str(tmp_path / "failing_migration.db")
        # Create table with only original columns so migration will attempt ALTER TABLE
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE analysis (id INTEGER PRIMARY KEY, url TEXT);")
        con.close()

        engine = create_db_engine(db_path)

        original_connect = engine.connect

        def faulty_connect(*args, **kwargs):
            conn = original_connect(*args, **kwargs)
            orig_execute = conn.execute

            def faulty_execute(statement, *exec_args, **exec_kwargs):
                sql_str = str(statement)
                if "ALTER TABLE analysis ADD COLUMN" in sql_str:
                    raise OperationalError("Simulated disk error during migration", {}, None)
                return orig_execute(statement, *exec_args, **exec_kwargs)

            conn.execute = faulty_execute
            return conn

        monkeypatch.setattr(engine, "connect", faulty_connect)

        with pytest.raises(DatabaseMigrationError) as exc_info:
            migrate_db(engine)

        assert "Failed to add column" in str(exc_info.value)
        engine.dispose()

    def test_refresh_views_failure_raises_descriptive_error(self, tmp_path, monkeypatch):
        """Requirement 8: Ensure view creation failure raises DatabaseMigrationError."""
        from sqlalchemy.exc import OperationalError

        db_path = str(tmp_path / "failing_view.db")
        engine = create_db_engine(db_path)
        Base.metadata.create_all(bind=engine)

        original_connect = engine.connect

        def faulty_connect(*args, **kwargs):
            conn = original_connect(*args, **kwargs)
            orig_execute = conn.execute

            def faulty_execute(statement, *exec_args, **exec_kwargs):
                sql_str = str(statement)
                if "CREATE VIEW analyses" in sql_str:
                    raise OperationalError("Simulated view creation failure", {}, None)
                return orig_execute(statement, *exec_args, **exec_kwargs)

            conn.execute = faulty_execute
            return conn

        monkeypatch.setattr(engine, "connect", faulty_connect)

        with pytest.raises(DatabaseMigrationError) as exc_info:
            refresh_views(engine)

        assert "Failed to create compatibility view 'analyses'" in str(exc_info.value)
        engine.dispose()

    def test_migrate_db_skips_when_analysis_table_not_present(self, tmp_path):
        """Verify migrate_db safely returns empty list when table does not exist."""
        db_path = str(tmp_path / "empty_db.db")
        engine = create_db_engine(db_path)
        # Table not created yet
        added = migrate_db(engine)
        assert added == []
        engine.dispose()


