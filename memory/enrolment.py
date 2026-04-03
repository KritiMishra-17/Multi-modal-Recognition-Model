"""
memory/enrolment.py — Safe + optimized face enrolment & recognition
"""

import os
import time
import cv2
import numpy as np

from memory.database import (
    find_best_face_match,
    load_all_face_embeddings,
    register_person,
    add_face_embedding,
    update_visit,
    get_angles_for_person,
    get_all_persons,
)

from face.pose import ALL_POSES

SIMILARITY_THRESHOLD    = 0.60
NEW_PERSON_COOLDOWN_SEC = 2.0
CAPTURE_DIR = "captures"

_last_new_time = 0
_emb_cache     = None
_pid_cache     = None
_cache_dirty   = True


def _invalidate_cache():
    global _cache_dirty
    _cache_dirty = True


def _get_cache():
    global _emb_cache, _pid_cache, _cache_dirty

    if _cache_dirty or _emb_cache is None:
        try:
            _emb_cache, _pid_cache, _ = load_all_face_embeddings()
        except Exception as e:
            print("[cache error]", e)
            _emb_cache, _pid_cache = [], []
        _cache_dirty = False

    return _emb_cache, _pid_cache


def reset_cache():
    _invalidate_cache()


def _save_capture(frame, person_id, angle):
    try:
        if frame is None or frame.size == 0:
            return ""

        folder = os.path.join(CAPTURE_DIR, person_id)
        os.makedirs(folder, exist_ok=True)

        ts   = int(time.time() * 1000)
        path = os.path.join(folder, f"{angle}_{ts}.jpg")

        cv2.imwrite(path, frame)
        return path

    except Exception as e:
        print("[capture error]", e)
        return ""


def _valid_embedding(embedding):
    if embedding is None:
        return False
    if not isinstance(embedding, np.ndarray):
        return False
    if embedding.size == 0:
        return False
    if np.isnan(embedding).any():
        return False
    if np.linalg.norm(embedding) == 0:
        return False
    return True


def enroll_face(embedding, frame=None, angle="unknown",
                blur_score=0.0, name=None):

    global _last_new_time

    # 🔴 Validate embedding
    if not _valid_embedding(embedding):
        return None, False, 0.0, ALL_POSES

    emb_cache, pid_cache = _get_cache()

    try:
        matched_id, score = find_best_face_match(
            embedding,
            threshold=SIMILARITY_THRESHOLD,
            embeddings=emb_cache,
            person_ids=pid_cache,
        )
    except Exception as e:
        print("[match error]", e)
        matched_id, score = None, 0.0

    # ── known person ─────────────────────────────
    if matched_id is not None:
        person_id = matched_id

        try:
            seen_angles = get_angles_for_person(person_id)
        except:
            seen_angles = set()

        if angle not in seen_angles:
            stored = add_face_embedding(person_id, angle, embedding, blur_score)
            if stored:
                _save_capture(frame, person_id, angle)
                _invalidate_cache()

        update_visit(person_id)

        remaining = ALL_POSES - get_angles_for_person(person_id)
        return person_id, False, score, remaining

    # ── cooldown guard ───────────────────────────
    now = time.time()
    if now - _last_new_time < NEW_PERSON_COOLDOWN_SEC:
        persons = get_all_persons()
        if persons:
            last_id = persons[-1]["person_id"]
            seen_angles = get_angles_for_person(last_id)
            return last_id, False, score, ALL_POSES - seen_angles

    # ── new person ───────────────────────────────
    _last_new_time = now

    try:
        if name:
            person_id = name.strip().replace(" ", "_")
        else:
            persons = get_all_persons()
            person_id = f"person_{len(persons)+1}"

        register_person(person_id)

        add_face_embedding(person_id, angle, embedding, blur_score)
        _save_capture(frame, person_id, angle)

        _invalidate_cache()

        print(f"[NEW PERSON] {person_id}  score={score:.3f}")

        return person_id, True, score, ALL_POSES - {angle}

    except Exception as e:
        print("[enroll error]", e)
        return None, False, score, ALL_POSES


