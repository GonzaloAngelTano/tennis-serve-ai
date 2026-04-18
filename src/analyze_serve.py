"""
analyze_serve.py – Tennis serve biomechanics analyser.

Runs MediaPipe Pose on a serve video, extracts joint angles and wrist height
frame-by-frame, overlays the skeleton and metrics in real time, and saves
the results to a CSV file for downstream analysis.

Usage:
    python src/analyze_serve.py
    python src/analyze_serve.py --video data/raw_videos/djokovic.mp4 --output outputs/djokovic_analysis.csv
    python src/analyze_serve.py --start 0 --end 1500
"""

import argparse
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ── Serve-phase thresholds (elbow angle in degrees) ───────────────────────────
READY_ELBOW_THRESHOLD   = 160   # arm extended, pre-toss
LOADING_ELBOW_THRESHOLD = 120   # trophy / loading position

# ── Default frame range to analyse ────────────────────────────────────────────
DEFAULT_START_FRAME = 0
DEFAULT_END_FRAME   = 3000      # ~100 s at 30 fps; covers a full rally segment

# ── MediaPipe handles ─────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ── Core helpers ──────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Return the angle at vertex B formed by the path A→B→C (0–180°).

    Args:
        a, b, c: Two-element [x, y] lists of normalised landmark coordinates.

    Returns:
        Angle in degrees as a float.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(np.degrees(radians))
    return 360 - angle if angle > 180 else angle


def classify_phase(elbow_angle):
    """Map elbow angle to a human-readable serve phase label."""
    if elbow_angle > READY_ELBOW_THRESHOLD:
        return "Ready"
    if elbow_angle > LOADING_ELBOW_THRESHOLD:
        return "Loading"
    return "Impact"


def extract_landmarks(landmarks):
    """Extract right-side arm and leg landmark coordinates into a dict."""
    def lm(name):
        p = landmarks[mp_pose.PoseLandmark[name].value]
        return [p.x, p.y]

    return {
        "shoulder": lm("RIGHT_SHOULDER"),
        "elbow":    lm("RIGHT_ELBOW"),
        "wrist":    lm("RIGHT_WRIST"),
        "hip":      lm("RIGHT_HIP"),
        "knee":     lm("RIGHT_KNEE"),
        "ankle":    lm("RIGHT_ANKLE"),
    }


def overlay_metrics(frame, lm_coords, angles, phase):
    """Draw skeleton overlay and angle annotations onto the frame in place."""
    h, w = frame.shape[:2]

    def to_px(pt):
        return tuple(np.multiply(pt, [w, h]).astype(int))

    annotations = [
        (lm_coords["elbow"],    f"Elbow: {int(angles['elbow'])}°",       (0, 255, 0)),
        (lm_coords["knee"],     f"Knee: {int(angles['knee'])}°",         (255, 255, 0)),
        (lm_coords["shoulder"], f"Shoulder: {int(angles['shoulder'])}°", (0, 200, 255)),
    ]
    for point, label, color in annotations:
        cv2.putText(frame, label, to_px(point),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Phase: {phase}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# ── Main analysis pipeline ────────────────────────────────────────────────────

def analyse_video(video_path, output_path, start=DEFAULT_START_FRAME,
                  end=DEFAULT_END_FRAME, display=True):
    """Run pose detection on a serve video and save per-frame angles to CSV.

    Args:
        video_path:  Path to the input video file.
        output_path: Destination path for the output CSV.
        start:       First frame index to process (inclusive).
        end:         Last frame index to process (inclusive).
        display:     Whether to show the live annotated window (default True).
                     Pass False for headless / scripted execution.

    Returns:
        pd.DataFrame with columns [frame, elbow_angle, knee_angle,
                                    wrist_height, shoulder_angle].

    Raises:
        FileNotFoundError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_count = start
    records = []

    print(f"Processing: {video_path}  (frames {start}–{end})")

    with mp_pose.Pose() as pose:
        while frame_count <= end:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.pose_landmarks:
                lm = extract_landmarks(result.pose_landmarks.landmark)

                angles = {
                    "elbow":    calculate_angle(lm["shoulder"], lm["elbow"],    lm["wrist"]),
                    "knee":     calculate_angle(lm["hip"],      lm["knee"],     lm["ankle"]),
                    "shoulder": calculate_angle(lm["hip"],      lm["shoulder"], lm["elbow"]),
                }
                phase = classify_phase(angles["elbow"])

                records.append([
                    frame_count,
                    angles["elbow"],
                    angles["knee"],
                    lm["wrist"][1],   # normalised y: 0 = top, 1 = bottom
                    angles["shoulder"],
                ])

                if display:
                    mp_drawing.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    overlay_metrics(frame, lm, angles, phase)

            if display:
                cv2.namedWindow("Serve Analysis", cv2.WINDOW_NORMAL)
                cv2.imshow("Serve Analysis", frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    df = pd.DataFrame(
        records,
        columns=["frame", "elbow_angle", "knee_angle", "wrist_height", "shoulder_angle"],
    )
    df.to_csv(output_path, index=False)
    print(f"Done — {len(df):,} frames analysed.")
    print(f"Results saved to: {output_path}")
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse a tennis serve video and extract biomechanics data."
    )
    parser.add_argument(
        "--video", default="data/raw_videos/amateur.mp4",
        help="Path to the input video file (default: amateur.mp4).",
    )
    parser.add_argument(
        "--output", default="outputs/amateur_analysis.csv",
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "--start", type=int, default=DEFAULT_START_FRAME,
        help=f"First frame to analyse (default: {DEFAULT_START_FRAME}).",
    )
    parser.add_argument(
        "--end", type=int, default=DEFAULT_END_FRAME,
        help=f"Last frame to analyse (default: {DEFAULT_END_FRAME}).",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headlessly — skip the live video window (useful for scripting).",
    )
    args = parser.parse_args()

    try:
        analyse_video(args.video, args.output, args.start, args.end,
                      display=not args.no_display)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
