
"""Small, transactional SQLite repository. Connections never cross requests."""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Database:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("DATABASE_PATH") or
                         Path(__file__).resolve().parents[2] / "data" / "autolens.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    id TEXT PRIMARY KEY, manufacturer TEXT NOT NULL, model TEXT NOT NULL,
                    year INTEGER NOT NULL, manual_filename TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    UNIQUE(manufacturer, model, year));
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, vehicle_id TEXT REFERENCES vehicles(id),
                    warning_type TEXT, severity TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]', severity_json TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS sessions_vehicle ON sessions(vehicle_id);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_vehicle(self, manufacturer, model, year, manual_filename=None):
        with self.connect() as db:
            db.execute("""INSERT INTO vehicles(id,manufacturer,model,year,manual_filename)
                VALUES(?,?,?,?,?) ON CONFLICT(manufacturer,model,year) DO UPDATE SET
                manual_filename=COALESCE(excluded.manual_filename,vehicles.manual_filename)""",
                       (str(uuid4()), manufacturer, model, year, manual_filename))
            return dict(db.execute("SELECT * FROM vehicles WHERE manufacturer=? AND model=? AND year=?",
                                   (manufacturer, model, year)).fetchone())

    def vehicle(self, vehicle_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM vehicles WHERE id=?", (vehicle_id,)).fetchone()
            return dict(row) if row else None

    def vehicles(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM vehicles ORDER BY created_at DESC")]

    def sessions(self, vehicle_id=None):
        with self.connect() as db:
            return [dict(r) for r in db.execute("""SELECT sessions.*,
                (SELECT content FROM messages WHERE session_id=sessions.id AND role='user'
                 ORDER BY id LIMIT 1) AS title FROM sessions
                WHERE (? IS NULL OR vehicle_id=?) ORDER BY updated_at DESC, id""", (vehicle_id, vehicle_id))]

    def session(self, session_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["messages"] = []
            for row in db.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)):
                message = dict(row)
                message["citations"] = json.loads(message.pop("citations_json"))
                message["severity"] = json.loads(message.pop("severity_json") or "null")
                result["messages"].append(message)
            return result

    def save_turn(self, session_id, vehicle_id, warning, message, result):
        """Save a complete turn atomically, so failures cannot leave half a transcript."""
        with self.connect() as db:
            db.execute("""INSERT INTO sessions(id,vehicle_id,warning_type,severity) VALUES(?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET warning_type=excluded.warning_type,
                severity=excluded.severity, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')""",
                       (session_id, vehicle_id, warning, result["severity"]["level"]))
            db.execute("INSERT INTO messages(session_id,role,content) VALUES(?,'user',?)", (session_id, message))
            db.execute("""INSERT INTO messages(session_id,role,content,citations_json,severity_json)
                VALUES(?,'assistant',?,?,?)""", (session_id, result["reply"],
                json.dumps(result["citations"]), json.dumps(result["severity"])))
