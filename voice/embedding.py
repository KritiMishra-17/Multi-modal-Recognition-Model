
import numpy as np
import traceback

# ── numpy compatibility shim ──────────────────────────────────────────────────
# Must happen BEFORE resemblyzer is imported. resemblyzer/webrtcvad uses
# np.bool, np.int etc. which were removed in NumPy 1.20+.
# We only patch attributes that are genuinely missing.
import builtins as _builtins

_np_compat = {
    "bool":    bool,
    "int":     int,
    "float":   float,
    "complex": complex,
    "object":  object,
    "str":     str,
}
for _k, _v in _np_compat.items():
    try:
        getattr(np, _k)          # already exists → skip (avoids FutureWarning)
    except AttributeError:
        setattr(np, _k, _v)

# Also silence the FutureWarnings that numpy itself emits when these are accessed
import warnings as _warnings
_warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

from resemblyzer import VoiceEncoder, preprocess_wav  # noqa: E402

encoder = VoiceEncoder()
print(f"[voice] encoder device: {encoder.device}")

_MIN_SAMPLES = 16000 * 2   # 2 s at 16 kHz — resemblyzer needs ~1 s minimum


def get_voice_embedding(path: str):
    """Return a 256-d np.ndarray embedding, or None on failure."""
    try:
        # preprocess_wav handles resampling and normalization internally.
        # Do NOT check raw amplitude after this — resemblyzer normalizes to [-1,1].
        wav = preprocess_wav(path)

        if wav is None or len(wav) == 0:
            print("[voice] preprocess_wav returned empty audio")
            return None

        # Pad short clips instead of rejecting them
        if len(wav) < _MIN_SAMPLES:
            wav = np.pad(wav, (0, _MIN_SAMPLES - len(wav)))

        emb = encoder.embed_utterance(wav)

        if emb is None or np.isnan(emb).any():
            print("[voice] invalid embedding produced")
            return None

        return emb

    except Exception:
        print("❌ Voice embedding crashed:")
        traceback.print_exc()
        return None