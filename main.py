
import cv2
import queue
import threading
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

from face.detector import FaceDetector
from face.embedder import FaceEmbedder
from face.pose import estimate_pose, pose_is_usable
from face.quality import face_is_usable, crop_face

from memory.enrolment import enroll_face, reset_cache
from memory.database import append_voice_embedding, find_best_voice_match

from voice.recorder import record_audio
from voice.embedding import get_voice_embedding


# ───────────────────────── CAMERA THREAD ─────────────────────────

def camera_thread(frame_q, event_q):
    detector = FaceDetector(ctx_id=0)
    embedder = FaceEmbedder()

    reset_cache()
    cap = cv2.VideoCapture(0)

    register_name = None

    while True:
        try:
            cmd = event_q.get_nowait()
            if cmd == "STOP":
                break
            if isinstance(cmd, dict) and cmd.get("cmd") == "REGISTER":
                register_name = cmd["name"]
        except queue.Empty:
            pass

        ret, frame = cap.read()
        if not ret:
            continue

        faces = detector.detect(frame)

        best_id    = None
        best_score = 0.0

        for face in faces:
            ok, _ = face_is_usable(frame, face)
            if not ok:
                continue

            emb = embedder.get_embedding(frame, face)
            if emb is None:
                continue

            pose = estimate_pose(face)
            if not pose_is_usable(pose):
                continue

            crop = crop_face(frame, face)
            blur = 0.0
            if crop is not None:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()

            person_id, _, score, _ = enroll_face(
                emb, frame, pose, blur, name=register_name
            )

            if register_name and person_id == register_name:
                register_name = None

            if score > best_score:
                best_id    = person_id
                best_score = score

            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{person_id} {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if frame_q.qsize() < 2:
            frame_q.put((frame, best_id, best_score))

    cap.release()


# ───────────────────────── GUI ─────────────────────────

class App:
    def __init__(self, root, frame_q, event_q):
        self.root     = root
        self.frame_q  = frame_q
        self.event_q  = event_q

        self.current_face_id  = None
        self.current_voice_id = None

        # ── widgets ──────────────────────────────────────
        self.video = tk.Label(root)
        self.video.pack()

        self.face_label = tk.Label(root, text="👤 Face: -", font=("Arial", 12))
        self.face_label.pack()

        self.voice_label = tk.Label(root, text="🎤 Voice: -", font=("Arial", 12))
        self.voice_label.pack()

        self.result_label = tk.Label(root, text="Result: -", font=("Arial", 14, "bold"))
        self.result_label.pack()

        tk.Button(root, text="📝 Register Face", command=self.register).pack()
        tk.Button(root, text="🎤 Add Voice",     command=self.add_voice).pack()
        tk.Button(root, text="🔍 Verify Voice",  command=self.verify_voice).pack()

        self.poll()

    # ───────────────── FRAME UPDATE ─────────────────

    def poll(self):
        try:
            frame, face_id, score = self.frame_q.get_nowait()

            self.current_face_id = face_id

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.config(image=img)
            self.video.image = img  # prevent GC

            if face_id:
                self.face_label.config(text=f"👤 Face: {face_id} ({score:.2f})")

        except queue.Empty:
            pass

        self.root.after(30, self.poll)

    # ───────────────── REGISTER FACE ─────────────────

    def register(self):
        name = simpledialog.askstring("Register", "Enter your name:")
        if not name:
            return
        self.event_q.put({"cmd": "REGISTER", "name": name.strip()})

    # ───────────────── ADD VOICE ─────────────────

    def add_voice(self):
        # Capture face_id RIGHT NOW at click time — not inside the thread,
        # where current_face_id may have changed or still be None.
        face_id = self.current_face_id
        if not face_id:
            self.result_label.config(text="❌ No face detected")
            return

        self.result_label.config(text="🎤 Recording voice...")
        threading.Thread(
            target=self._add_voice,
            args=(face_id,),
            daemon=True
        ).start()

    def _add_voice(self, face_id: str):
        path = record_audio()
        emb  = get_voice_embedding(path)

        def update_ui():
            if emb is None:
                self.result_label.config(text="❌ Voice capture failed")
                return
            append_voice_embedding(face_id, emb)
            self.voice_label.config(text=f"🎤 Voice added for {face_id}")
            self.result_label.config(text=f"✅ Voice enrolled for {face_id}")

        self.root.after(0, update_ui)

    # ───────────────── VERIFY VOICE ─────────────────

    def verify_voice(self):
        # Capture face_id at click time for the comparison at the end.
        face_id = self.current_face_id
        self.voice_label.config(text="🎤 Recording...")
        threading.Thread(
            target=self._verify_voice,
            args=(face_id,),
            daemon=True
        ).start()

    def _verify_voice(self, face_id: str):
        path = record_audio()
        emb  = get_voice_embedding(path)

        def update_ui():
            if emb is None:
                self.voice_label.config(text="❌ Voice capture failed")
                return

            voice_id, score, confidence = find_best_voice_match(emb)
            self.current_voice_id = voice_id

            if voice_id:
                self.voice_label.config(
                    text=f"🎤 Voice: {voice_id} ({score:.2f}) [{confidence}]"
                )
            else:
                self.voice_label.config(text="🎤 Voice: unknown")

            if face_id and face_id == voice_id:
                self.result_label.config(text=f"✅ VERIFIED — {voice_id}")
            else:
                self.result_label.config(
                    text=f"❌ MISMATCH  face={face_id or '?'}  voice={voice_id or '?'}"
                )

        self.root.after(0, update_ui)


# ───────────────────────── MAIN ─────────────────────────

def main():
    frame_q = queue.Queue(maxsize=5)
    event_q = queue.Queue()

    threading.Thread(
        target=camera_thread,
        args=(frame_q, event_q),
        daemon=True
    ).start()

    root = tk.Tk()
    root.title("Biometric System")

    App(root, frame_q, event_q)

    def on_close():
        event_q.put("STOP")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()