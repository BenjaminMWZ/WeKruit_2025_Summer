import cv2, csv, mediapipe as mp
from tqdm import tqdm

VIDEO_PATH = "/Users/benjaminmao/Desktop/Rice/2025Summer/WeKruit/interview_recording.mp4"
OUT_CSV    = "iris-gaze.csv"

# ─── 1. Set up FaceMesh with iris refinement ──────────────────────────────
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ─── 2. Open video & get frame count for progress bar ─────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ─── 3. Loop with tqdm progress bar ───────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f, tqdm(total=total_frames,
                                               desc="Processing frames",
                                               unit="frame") as pbar:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "left_x","left_y","left_z",
        "right_x","right_y","right_z"
    ])

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            L, R = lm[468], lm[473]    # iris centres
            writer.writerow([
                frame_idx,
                L.x, L.y, L.z,
                R.x, R.y, R.z
            ])

        frame_idx += 1
        pbar.update(1)                 # advance progress bar

cap.release()
face_mesh.close()
print(f"\nDone! CSV written to {OUT_CSV}")