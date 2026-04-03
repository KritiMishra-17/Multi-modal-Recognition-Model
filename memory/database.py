"""
memory/database.py — Thread-safe SQLite face+voice biometric store
WAL mode, per-thread connections, RLock for writes.
"""

import os
import json
import sqlite3
import threading
import numpy as np
from datetime import datetime

DB_PATH = os.path.join("store", "biometric.db")

_local = threading.local()
_lock  = threading.RLock()

MAX_EMB_PER_ANGLE  = 5
MAX_EMB_PER_PERSON = 30
SIMILARITY_THRESHOLD = 0.60    # InceptionResnetV1 (vggface2) cosine similarity


# ─── connection ──────────────────────────────────────────────────────────────

def _conn():
    if not hasattr(_local, "conn"):
        os.makedirs("store", exist_ok=True)
        c = sqlite3.connect(DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        _local.conn = c
        _init_schema(c)
    return _local.conn


def _init_schema(c):
    c.executescript("""
    CREATE TABLE IF NOT EXISTS persons (
        person_id    TEXT PRIMARY KEY,
        display_name TEXT,
        created_at   TEXT,
        first_seen   TEXT,
        last_seen    TEXT,
        visit_count  INTEGER DEFAULT 0,
        voice_blob   BLOB    -- pickled list of voice embeddings
    );

    CREATE TABLE IF NOT EXISTS face_embeddings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id   TEXT REFERENCES persons(person_id),
        angle       TEXT,
        vector      BLOB,
        captured_at TEXT,
        blur_score  REAL
    );

    CREATE INDEX IF NOT EXISTS idx_face_person
    ON face_embeddings(person_id);
    """)
    c.commit()


# ─── vector helpers ───────────────────────────────────────────────────────────

def _vec_to_blob(v):
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.astype(np.float32).tobytes()


def _blob_to_vec(b):
    return np.frombuffer(b, dtype=np.float32).copy()


# ─── person CRUD ─────────────────────────────────────────────────────────────

def get_all_persons():
    rows = _conn().execute(
        "SELECT * FROM persons ORDER BY person_id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_person(pid):
    row = _conn().execute(
        "SELECT * FROM persons WHERE person_id=?", (pid,)
    ).fetchone()
    return dict(row) if row else None


def register_person(pid, display_name=""):
    with _lock:
        now = datetime.now().isoformat(timespec="seconds")
        try:
            _conn().execute(
                """INSERT INTO persons
                   (person_id, display_name, created_at, first_seen, last_seen)
                   VALUES (?,?,?,?,?)""",
                (pid, display_name or pid, now, now, now),
            )
            _conn().commit()
            return True
        except sqlite3.IntegrityError:
            return False


def update_visit(pid):
    with _lock:
        now = datetime.now().isoformat(timespec="seconds")
        _conn().execute(
            """UPDATE persons SET visit_count=visit_count+1, last_seen=?
               WHERE person_id=?""",
            (now, pid),
        )
        _conn().commit()


def rename_person_db(old_id, new_id):
    with _lock:
        c = _conn()
        try:
            c.execute("UPDATE face_embeddings SET person_id=? WHERE person_id=?", (new_id, old_id))
            c.execute("UPDATE persons SET person_id=? WHERE person_id=?", (new_id, old_id))
            c.commit()
            return True
        except Exception as e:
            print("[db] rename failed:", e)
            c.rollback()
            return False


def delete_person_db(pid):
    with _lock:
        c = _conn()
        c.execute("DELETE FROM face_embeddings WHERE person_id=?", (pid,))
        c.execute("DELETE FROM persons WHERE person_id=?", (pid,))
        c.commit()


def clear_all_db():
    with _lock:
        c = _conn()
        c.execute("DELETE FROM face_embeddings")
        c.execute("DELETE FROM persons")
        c.commit()


# ─── face embeddings ─────────────────────────────────────────────────────────

def get_angles_for_person(pid):
    rows = _conn().execute(
        "SELECT DISTINCT angle FROM face_embeddings WHERE person_id=?", (pid,)
    ).fetchall()
    return {r["angle"] for r in rows}


def count_face_embeddings(pid):
    row = _conn().execute(
        "SELECT COUNT(*) as n FROM face_embeddings WHERE person_id=?", (pid,)
    ).fetchone()
    return row["n"]


def load_all_face_embeddings():
    rows = _conn().execute(
        "SELECT person_id, angle, vector FROM face_embeddings"
    ).fetchall()
    if not rows:
        return [], [], []
    embs, pids, angles = [], [], []
    for r in rows:
        embs.append(_blob_to_vec(r["vector"]))
        pids.append(r["person_id"])
        angles.append(r["angle"])
    return embs, pids, angles


def add_face_embedding(pid, angle, embedding, blur_score=0.0):
    with _lock:
        c = _conn()
        total = count_face_embeddings(pid)
        if total >= MAX_EMB_PER_PERSON:
            return False
        rows = c.execute(
            "SELECT id FROM face_embeddings WHERE person_id=? AND angle=? ORDER BY id ASC",
            (pid, angle),
        ).fetchall()
        if len(rows) >= MAX_EMB_PER_ANGLE:
            c.execute("DELETE FROM face_embeddings WHERE id=?", (rows[0]["id"],))
        c.execute(
            """INSERT INTO face_embeddings (person_id,angle,vector,captured_at,blur_score)
               VALUES (?,?,?,?,?)""",
            (pid, angle, _vec_to_blob(embedding),
             datetime.now().isoformat(timespec="seconds"), blur_score),
        )
        c.commit()
        return True


def find_best_face_match(query, threshold=SIMILARITY_THRESHOLD,
                         embeddings=None, person_ids=None):
    if embeddings is None:
        embeddings, person_ids, _ = load_all_face_embeddings()
    if not embeddings:
        return None, 0.0

    query = query / (np.linalg.norm(query) + 1e-9)
    mat   = np.array(embeddings, dtype=np.float32)
    mat  /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    scores = mat @ query

    best_scores = {}
    for pid, score in zip(person_ids, scores):
        if pid not in best_scores or score > best_scores[pid]:
            best_scores[pid] = float(score)

    best_pid   = max(best_scores, key=best_scores.get)
    best_score = best_scores[best_pid]
    return (best_pid, best_score) if best_score >= threshold else (None, best_score)


# ─── voice embeddings (stored as pickle blob per person) ─────────────────────

import pickle

VOICE_THRESHOLD = 0.65
VOICE_MIN_DIFF  = 0.08
MAX_VOICE_SAMPLES = 10


def save_voice_embeddings(pid, embeddings: list):
    with _lock:
        blob = pickle.dumps(embeddings)
        _conn().execute(
            "UPDATE persons SET voice_blob=? WHERE person_id=?", (blob, pid)
        )
        _conn().commit()


def load_voice_embeddings(pid):
    row = _conn().execute(
        "SELECT voice_blob FROM persons WHERE person_id=?", (pid,)
    ).fetchone()
    if row is None or row["voice_blob"] is None:
        return []
    return pickle.loads(row["voice_blob"])


def append_voice_embedding(pid, new_emb):
    embs = load_voice_embeddings(pid)
    if not isinstance(embs, list):
        embs = [embs]
    embs.append(new_emb)
    embs = embs[-MAX_VOICE_SAMPLES:]
    save_voice_embeddings(pid, embs)


def find_best_voice_match(new_emb):
    """Returns (person_id, score, confidence) or (None, score, 'LOW')."""
    persons = get_all_persons()
    if not persons:
        return None, 0.0, "LOW"

    scores_dict = {}
    for p in persons:
        stored = load_voice_embeddings(p["person_id"])
        if not stored:
            continue
        sims = []
        for emb in stored:
            a = np.array(new_emb).reshape(1, -1)
            b = np.array(emb).reshape(1, -1)
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            sims.append(cos_sim(a, b)[0][0])
        scores_dict[p["person_id"]] = sum(sims) / len(sims)

    if not scores_dict:
        return None, 0.0, "LOW"

    sorted_users = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    best_user, best_score = sorted_users[0]
    second_score = sorted_users[1][1] if len(sorted_users) > 1 else 0.0
    diff = best_score - second_score

    if best_score < VOICE_THRESHOLD:
        return None, best_score, "LOW"
    if diff < VOICE_MIN_DIFF:
        return best_user, best_score, "LOW"
    return best_user, best_score, "HIGH"

