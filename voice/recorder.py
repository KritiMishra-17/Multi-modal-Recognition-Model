
import os
import subprocess

import numpy as np
import soundfile as sf


def _record_via_arecord(path: str, duration: int = 6, fs: int = 16000) -> bool:
    """Use arecord (ALSA) — more stable than sounddevice on Linux threads."""
    try:
        cmd = [
            "arecord",
            "-f", "S16_LE",
            "-r", str(fs),
            "-c", "1",
            "-d", str(duration),
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=duration + 5)
        return result.returncode == 0 and os.path.exists(path)
    except Exception as e:
        print(f"[recorder] arecord failed: {e}")
        return False


def _record_via_sounddevice(path: str, duration: int = 6, fs: int = 16000) -> bool:
    try:
        import sounddevice as sd
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        sf.write(path, audio, fs)
        return True
    except Exception as e:
        print(f"[recorder] sounddevice failed: {e}")
        return False


def record_audio(duration: int = 6, fs: int = 16000) -> str:
    print("🎤 Recording... Speak now!")

    path = "temp.wav"

    # Try sounddevice first; fall back to arecord on Linux crash
    success = _record_via_sounddevice(path, duration, fs)
    if not success:
        print("[recorder] falling back to arecord...")
        success = _record_via_arecord(path, duration, fs)

    if not success or not os.path.exists(path):
        print("[recorder] ❌ recording failed entirely")
        return path

    # Load back for trimming
    audio, actual_fs = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Conservative threshold — avoids over-trimming quiet speakers
    threshold = 0.005
    indices = np.where(np.abs(audio) > threshold)[0]

    if len(indices) > 0:
        start = max(0, indices[0] - int(0.15 * actual_fs))
        end   = min(len(audio), indices[-1] + int(0.15 * actual_fs))
        audio = audio[start:end]
        print(f"[recorder] trimmed to {len(audio) / actual_fs:.2f}s of speech")
    else:
        print("[recorder] ⚠️  no speech above threshold — keeping full clip")

    # Pad to at least 2 s so resemblyzer never rejects the clip
    min_samples = int(2.0 * actual_fs)
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))

    sf.write(path, audio, actual_fs)
    return path