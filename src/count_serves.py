import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose

# Calibrated thresholds (from inspecting landmark values in the actual videos)
# wrist_diff = wrist.y - shoulder.y  →  negative means wrist is ABOVE shoulder
RAISE_THRESHOLD = 0.07   # arm is "raised" when best_diff < -0.07
DROP_THRESHOLD  = 0.03   # arm is "down" when diff > -0.03 (back near or below shoulder)
MIN_RAISE_FRAMES = 5     # ignore blips shorter than this
MAX_RAISED_FRAMES = 300  # safety timeout in case arm never fully comes down
COOLDOWN_FRAMES = 90     # ~3s at 30fps — skip after each serve to avoid double-counting


def _best_diff(lm):
    """Return the most-negative wrist-shoulder diff across both arms (most raised)."""
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    return min(r_wr - r_sh, l_wr - l_sh)  # negative = wrist above shoulder


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    serve_count = 0
    state = 0          # 0=idle, 1=arm_raised
    frames_in_state = 0
    cooldown = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if cooldown > 0:
                cooldown -= 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                if state == 1:
                    frames_in_state += 1
                continue

            diff = _best_diff(results.pose_landmarks.landmark)
            arm_up = diff < -RAISE_THRESHOLD

            if state == 0:
                if arm_up:
                    state = 1
                    frames_in_state = 1

            elif state == 1:
                if arm_up:
                    frames_in_state += 1
                    if frames_in_state > MAX_RAISED_FRAMES:
                        # Timeout — count it anyway if arm was up long enough
                        if frames_in_state >= MIN_RAISE_FRAMES:
                            serve_count += 1
                            cooldown = COOLDOWN_FRAMES
                        state = 0
                        frames_in_state = 0
                else:
                    # Arm came back down
                    if frames_in_state >= MIN_RAISE_FRAMES:
                        serve_count += 1
                        cooldown = COOLDOWN_FRAMES
                    state = 0
                    frames_in_state = 0

    cap.release()
    return serve_count, total_frames, fps, duration


VIDEOS = {
    "djokovic.mp4": "data/raw_videos/djokovic.mp4",
    "federer.mp4":  "data/raw_videos/federer.mp4",
    "amateur.mp4":  "data/raw_videos/amateur.mp4",
}

for name, path in VIDEOS.items():
    if not os.path.exists(path):
        print(f"\n{name}: archivo no encontrado ({path})")
        continue

    result = analyze_video(path)
    if result is None:
        print(f"\n{name}: no se pudo abrir el video")
        continue

    serves, frames, fps, duration = result
    print(f"\n{name}:")
    print(f"  Saques detectados : {serves}")
    print(f"  Total frames      : {frames}")
    print(f"  FPS               : {fps:.1f}")
    print(f"  Duracion (seg)    : {duration:.1f}")
