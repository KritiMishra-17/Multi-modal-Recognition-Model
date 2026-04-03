
"""
voice/recognizer.py — Vosk offline speech-to-text.

Usage:
    from voice.recognizer import transcribe

Place the Vosk model folder (e.g. vosk-model-small-en-us-0.15) inside
the project root and set VOSK_MODEL_PATH accordingly, or pass model_path
to transcribe().
"""

import json
import os
import threading
import wave

VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")

_model      = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:   # double-checked locking
            return _model
        try:
            from vosk import Model
            if not os.path.exists(VOSK_MODEL_PATH):
                print(f"[vosk] model not found at {VOSK_MODEL_PATH!r} — STT disabled")
                return None
            _model = Model(VOSK_MODEL_PATH)
            print("[vosk] model loaded ✓")
        except ImportError:
            print("[vosk] package not installed — STT disabled")
    return _model


def transcribe(audio_path: str) -> str:
    """Return transcribed text, or empty string on failure."""
    if not audio_path or not os.path.exists(audio_path):
        return ""

    model = _get_model()
    if model is None:
        return ""

    wf = None
    try:
        from vosk import KaldiRecognizer
        wf  = wave.open(audio_path, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        text_parts = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text_parts.append(res.get("text", ""))

        final = json.loads(rec.FinalResult())
        text_parts.append(final.get("text", ""))

        return " ".join(t for t in text_parts if t).strip()

    except Exception as e:
        print(f"[vosk] transcription error: {e}")
        return ""
    finally:
        if wf is not None:
            wf.close()