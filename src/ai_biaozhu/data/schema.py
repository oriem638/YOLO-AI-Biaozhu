"""SQLite connection setup and forward-only schema initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ai_biaozhu.core.domain import PROJECT_SCHEMA_VERSION
from ai_biaozhu.core.exceptions import ProjectFormatError

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS categories (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT,
    color TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS category_name_aliases (
    alias TEXT PRIMARY KEY COLLATE NOCASE,
    category_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_category_name_aliases_category
ON category_name_aliases(category_id, alias);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    original_name TEXT NOT NULL,
    source_path TEXT,
    sha256 TEXT NOT NULL UNIQUE CHECK (length(sha256) = 64),
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    review_status TEXT NOT NULL DEFAULT 'unreviewed'
        CHECK (review_status IN ('unreviewed', 'draft', 'verified')),
    origin TEXT NOT NULL DEFAULT 'none'
        CHECK (origin IN ('none', 'manual', 'ai', 'mixed')),
    ai_status TEXT NOT NULL DEFAULT 'none'
        CHECK (ai_status IN ('none', 'queued', 'running', 'ready', 'failed')),
    training_selected INTEGER NOT NULL DEFAULT 1
        CHECK (training_selected IN (0, 1)),
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    imported_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_review_status
ON images(review_status, imported_at, id);

CREATE INDEX IF NOT EXISTS idx_images_ai_status
ON images(ai_status, imported_at, id);

CREATE INDEX IF NOT EXISTS idx_images_training_selected
ON images(training_selected, review_status, imported_at, id);

CREATE TABLE IF NOT EXISTS model_runs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('train', 'predict', 'deploy')),
    model_key TEXT NOT NULL,
    status TEXT NOT NULL,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    snapshot_path TEXT,
    metrics_jsonl_path TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    artifacts_json TEXT NOT NULL DEFAULT '{}',
    checkpoint_path TEXT,
    progress REAL NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_runs_kind_status
ON model_runs(kind, status, created_at);

CREATE TABLE IF NOT EXISTS boxes (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class_id TEXT NOT NULL REFERENCES categories(id) ON DELETE RESTRICT,
    x1 REAL NOT NULL CHECK (x1 >= 0),
    y1 REAL NOT NULL CHECK (y1 >= 0),
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('manual', 'ai', 'mixed')),
    confidence REAL CHECK (
        confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
    ),
    model_run_id TEXT REFERENCES model_runs(id) ON DELETE SET NULL,
    prediction_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (x1 < x2),
    CHECK (y1 < y2)
);

CREATE INDEX IF NOT EXISTS idx_boxes_image
ON boxes(image_id, created_at, id);

CREATE INDEX IF NOT EXISTS idx_boxes_class
ON boxes(class_id, image_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_boxes_prediction_idempotency
ON boxes(model_run_id, image_id, prediction_id)
WHERE model_run_id IS NOT NULL AND prediction_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES model_runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('train', 'predict', 'deploy', 'environment')),
    status TEXT NOT NULL,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    last_seq INTEGER NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS image_job_results (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (job_id, image_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS ai_imports (
    run_id TEXT NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    box_count INTEGER NOT NULL CHECK (box_count >= 0),
    imported_at TEXT NOT NULL,
    PRIMARY KEY (run_id, image_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS deployment_packages (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    target TEXT NOT NULL,
    checkpoint_role TEXT NOT NULL,
    npu_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    model_package_path TEXT,
    app_package_path TEXT,
    report_path TEXT,
    zip_bytes INTEGER CHECK (zip_bytes IS NULL OR zip_bytes >= 0),
    payload_bytes INTEGER CHECK (payload_bytes IS NULL OR payload_bytes >= 0),
    warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deployment_packages_run
ON deployment_packages(run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_deployment_packages_target_status
ON deployment_packages(target, status, created_at);
"""


def connect_database(path: Path, *, initialize: bool = True) -> sqlite3.Connection:
    """Open a configured SQLite connection suitable as the editing source of truth."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path,
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    if initialize:
        try:
            initialize_schema(connection)
        except Exception:
            connection.close()
            raise
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > PROJECT_SCHEMA_VERSION:
        raise ProjectFormatError(
            f"数据库版本 {current} 高于软件支持的版本 {PROJECT_SCHEMA_VERSION}"
        )
    if current == 0:
        try:
            connection.executescript(
                "BEGIN IMMEDIATE;\n"
                + SCHEMA_SQL
                + f"\nPRAGMA user_version = {PROJECT_SCHEMA_VERSION};\n"
                + "COMMIT;\n"
            )
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        return
    if current == 1:
        _migrate_v1_to_v2(connection)
        current = 2
    if current == 2:
        _migrate_v2_to_v3(connection)
        current = 3
    if current == 3:
        _migrate_v3_to_v4(connection)
        current = 4
    if current == 4:
        _migrate_v4_to_v5(connection)
        current = 5
    if current == PROJECT_SCHEMA_VERSION:
        # CREATE IF NOT EXISTS also repairs optional indexes within this version.
        connection.executescript(SCHEMA_SQL)
    else:
        raise ProjectFormatError(f"暂不支持从数据库版本 {current} 迁移")


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    """Rebuild the two CHECK-constrained tables and add deployment packages."""

    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.executescript(
            """
            BEGIN IMMEDIATE;

            CREATE TABLE model_runs_v2 (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('train', 'predict', 'deploy')),
                model_key TEXT NOT NULL,
                status TEXT NOT NULL,
                parameters_json TEXT NOT NULL DEFAULT '{}',
                snapshot_path TEXT,
                metrics_jsonl_path TEXT,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                artifacts_json TEXT NOT NULL DEFAULT '{}',
                checkpoint_path TEXT,
                progress REAL NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );

            INSERT INTO model_runs_v2
            SELECT * FROM model_runs;

            CREATE TABLE jobs_v2 (
                id TEXT PRIMARY KEY,
                run_id TEXT REFERENCES model_runs(id) ON DELETE CASCADE,
                kind TEXT NOT NULL
                    CHECK (kind IN ('train', 'predict', 'deploy', 'environment')),
                status TEXT NOT NULL,
                parameters_json TEXT NOT NULL DEFAULT '{}',
                last_seq INTEGER NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                error TEXT
            );

            INSERT INTO jobs_v2
            SELECT * FROM jobs;

            DROP TABLE jobs;
            DROP TABLE model_runs;
            ALTER TABLE model_runs_v2 RENAME TO model_runs;
            ALTER TABLE jobs_v2 RENAME TO jobs;

            CREATE INDEX idx_model_runs_kind_status
            ON model_runs(kind, status, created_at);

            CREATE INDEX idx_jobs_status ON jobs(status, created_at);

            CREATE TABLE IF NOT EXISTS deployment_packages (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
                target TEXT NOT NULL,
                checkpoint_role TEXT NOT NULL,
                npu_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                model_package_path TEXT,
                app_package_path TEXT,
                report_path TEXT,
                zip_bytes INTEGER CHECK (zip_bytes IS NULL OR zip_bytes >= 0),
                payload_bytes INTEGER CHECK (payload_bytes IS NULL OR payload_bytes >= 0),
                warnings_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_deployment_packages_run
            ON deployment_packages(run_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_deployment_packages_target_status
            ON deployment_packages(target, status, created_at);

            PRAGMA user_version = 2;
            COMMIT;
            """
        )
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
    violation = connection.execute("PRAGMA foreign_key_check").fetchone()
    if violation is not None:
        raise ProjectFormatError(f"数据库 v1→v2 迁移后外键校验失败：{tuple(violation)}")


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    """Add the persisted per-image training-selection flag.

    Existing projects intentionally default every image to selected so opening
    an older project never silently changes the dataset used for training.
    """

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(images)").fetchall()
    }
    add_column = (
        """
        ALTER TABLE images
        ADD COLUMN training_selected INTEGER NOT NULL DEFAULT 1
            CHECK (training_selected IN (0, 1));
        """
        if "training_selected" not in columns
        else ""
    )
    try:
        connection.executescript(
            "BEGIN IMMEDIATE;\n"
            + add_column
            + """
            CREATE INDEX IF NOT EXISTS idx_images_training_selected
            ON images(training_selected, review_status, imported_at, id);
            PRAGMA user_version = 3;
            COMMIT;
            """
        )
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    """Add an optional display/deployment alias for every canonical class."""

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(categories)").fetchall()
    }
    add_column = (
        "ALTER TABLE categories ADD COLUMN display_name TEXT;"
        if "display_name" not in columns
        else ""
    )
    try:
        connection.executescript(
            "BEGIN IMMEDIATE;\n"
            + add_column
            + "\nPRAGMA user_version = 4;\nCOMMIT;\n"
        )
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    """Persist historical canonical names used for compatible VOC imports.

    Version 4 projects already contain the optional display-name column, but
    predate the alias table used by full canonical category renames.  Keep the
    migration idempotent so projects briefly opened by an affected 0.2.2
    development build (which could create the table without advancing the
    schema version) are upgraded safely as well.
    """

    try:
        connection.executescript(
            """
            BEGIN IMMEDIATE;

            CREATE TABLE IF NOT EXISTS category_name_aliases (
                alias TEXT PRIMARY KEY COLLATE NOCASE,
                category_id TEXT NOT NULL
                    REFERENCES categories(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_category_name_aliases_category
            ON category_name_aliases(category_id, alias);

            PRAGMA user_version = 5;
            COMMIT;
            """
        )
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
