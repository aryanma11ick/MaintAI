"""SQLite is the local source of truth; model activation is one atomic commit."""
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from . import config as c

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@contextmanager
def db():
    c.ensure_dirs()
    conn = sqlite3.connect(c.ROOT / "maintai.db", timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with db() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS models (
          version TEXT PRIMARY KEY, path TEXT NOT NULL, digest TEXT NOT NULL,
          metrics TEXT NOT NULL, created REAL NOT NULL, status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
          kind TEXT NOT NULL, details TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions (
          id TEXT PRIMARY KEY, ts REAL NOT NULL, features TEXT NOT NULL,
          score REAL NOT NULL, predicted INTEGER NOT NULL, model TEXT NOT NULL,
          source_id INTEGER UNIQUE, scenario TEXT NOT NULL,
          labels TEXT, pending_labels TEXT, label_due REAL, consumed INTEGER NOT NULL DEFAULT 0);
        ''')

def get(key, default=None):
    with db() as con:
        row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default

def put(key, value):
    with db() as con:
        con.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))

def event(kind, details):
    with db() as con:
        con.execute("INSERT INTO events(ts,kind,details) VALUES (?,?,?)", (time.time(),kind,json.dumps(details)))

def active_record():
    with db() as con:
        row = con.execute("SELECT m.* FROM models m JOIN kv k ON k.key='active_model' AND m.version=json_extract(k.value,'$')").fetchone()
    if row is None:
        raise RuntimeError("No active model. Run: python -m maintai bootstrap")
    return dict(row)

def activate(version, reason, expected=None):
    """A compare-and-swap prevents a stale worker replacing a newer champion."""
    with db() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT value FROM kv WHERE key='active_model'").fetchone()
        old = json.loads(row[0]) if row else None
        if expected is not None and old != expected:
            raise RuntimeError("Champion changed during evaluation; refusing activation")
        target = con.execute("SELECT * FROM models WHERE version=?", (version,)).fetchone()
        if target is None or target['status'] in ('rejected', 'broken'):
            raise ValueError("Model is unavailable for activation")
        if old:
            con.execute("INSERT OR REPLACE INTO kv VALUES ('previous_model',?)", (json.dumps(old),))
            con.execute("UPDATE models SET status='retired' WHERE version=? AND status!='broken'",(old,))
        con.execute("UPDATE models SET status='active' WHERE version=?",(version,))
        con.execute("INSERT OR REPLACE INTO kv VALUES ('active_model',?)", (json.dumps(version),))
        con.execute("INSERT INTO events(ts,kind,details) VALUES (?,?,?)",(time.time(),'model_activated',json.dumps({'from':old,'to':version,'reason':reason})))

def recent_events(limit=50):
    with db() as con:
        rows = con.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
    return [{**dict(r),'details':json.loads(r['details'])} for r in rows]
