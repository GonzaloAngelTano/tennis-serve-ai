"""
multi_serve_analysis.py
-----------------------
Detects every serve in three tennis videos, extracts per-frame biomechanics
(elbow / knee / shoulder angle, wrist height, score), saves one CSV per serve,
then produces summary statistics and box-plot comparisons.

Serve detection strategy (two-stage):
  1. Wrist-above-shoulder state machine (same logic that correctly found
     9/2 serves in djokovic/amateur) marks each serve interval.
  2. Within each interval the frame with the highest impact score is chosen
     as the exact impact moment:
         score = elbow_angle + knee_angle - 300 * wrist_height

Usage:
    python src/multi_serve_analysis.py
"""

import os

import cv2
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

# ── Video paths ───────────────────────────────────────────────────────────────
VIDEOS = {
    "djokovic": "data/raw_videos/djokovic.mp4",
    "federer":  "data/raw_videos/test_video.mp4",
    "amateur":  "data/raw_videos/amateur.mp4",
}

# ── Wrist-detection thresholds (calibrated from debug inspection) ─────────────
# wrist_diff = wrist.y - shoulder.y  (negative = wrist above shoulder)
RAISE_THRESHOLD  = 0.07   # arm "raised" when best_diff < -0.07
MIN_RAISE_FRAMES = 5      # ignore blips shorter than this
MAX_RAISED_FRAMES = 300   # safety timeout (10 s at 30 fps)
COOLDOWN_FRAMES  = 90     # skip ~3 s after each serve to avoid double-count

# ── Serve window around impact ────────────────────────────────────────────────
PRE_IMPACT_FRAMES  = 25   # frames before impact kept in each serve CSV
POST_IMPACT_FRAMES = 10   # frames after impact

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Angle at vertex B in the A->B->C path, returned in degrees (0-180)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(np.degrees(radians))
    return 360 - angle if angle > 180 else angle


def get_xy(landmarks, name):
    """Return [x, y] normalised coords for a named MediaPipe landmark."""
    p = landmarks[mp_pose.PoseLandmark[name].value]
    return [p.x, p.y]


def extract_frame_metrics(landmarks):
    """
    Compute joint angles, wrist height, impact score, and the wrist-shoulder
    differential used for serve detection.

    Returns a dict with keys:
        elbow_angle, knee_angle, shoulder_angle,
        wrist_height, score, best_diff
    """
    r_sh  = get_xy(landmarks, "RIGHT_SHOULDER")
    r_el  = get_xy(landmarks, "RIGHT_ELBOW")
    r_wr  = get_xy(landmarks, "RIGHT_WRIST")
    r_hip = get_xy(landmarks, "RIGHT_HIP")
    r_kn  = get_xy(landmarks, "RIGHT_KNEE")
    r_an  = get_xy(landmarks, "RIGHT_ANKLE")
    l_sh  = get_xy(landmarks, "LEFT_SHOULDER")
    l_wr  = get_xy(landmarks, "LEFT_WRIST")

    elbow_angle    = calculate_angle(r_sh, r_el, r_wr)
    knee_angle     = calculate_angle(r_hip, r_kn, r_an)
    shoulder_angle = calculate_angle(r_hip, r_sh, r_el)

    # Use the highest wrist (smallest y) across both arms for the score.
    # This makes detection work for both right- and left-handed players.
    wrist_height = min(r_wr[1], l_wr[1])
    score = elbow_angle + knee_angle - 300 * wrist_height

    # best_diff: how far the more-raised wrist is above its shoulder.
    # Negative means wrist is above shoulder.
    r_diff = r_wr[1] - r_sh[1]
    l_diff = l_wr[1] - l_sh[1]
    best_diff = min(r_diff, l_diff)

    return {
        "elbow_angle":    elbow_angle,
        "knee_angle":     knee_angle,
        "shoulder_angle": shoulder_angle,
        "wrist_height":   wrist_height,
        "score":          score,
        "best_diff":      best_diff,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Video processing
# ─────────────────────────────────────────────────────────────────────────────

def process_video(video_path):
    """
    Run MediaPipe Pose on every frame; return a DataFrame with per-frame
    metrics, plus fps and total_frames metadata.
    """
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    records   = []
    frame_idx = 0

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
            frame_idx += 1

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                metrics = extract_frame_metrics(result.pose_landmarks.landmark)
                metrics["frame"] = frame_idx
                records.append(metrics)

            if frame_idx % 300 == 0:
                print(f"    frame {frame_idx} / {total_frames}")

    cap.release()
    # Reorder columns so 'frame' comes first
    df = pd.DataFrame(records)
    cols = ["frame", "elbow_angle", "knee_angle", "shoulder_angle",
            "wrist_height", "score", "best_diff"]
    return df[cols], fps, total_frames


# ─────────────────────────────────────────────────────────────────────────────
# Serve detection  (wrist-above-shoulder state machine)
# ─────────────────────────────────────────────────────────────────────────────

def find_serve_intervals(df):
    """
    Scan the best_diff column with the same state machine used in
    count_serves.py (which correctly found 9/2 serves in djokovic/amateur).

    Returns a list of (raise_start_pos, arm_down_pos) pairs of positional
    indices into df.  Each pair is one serve interval.
    """
    best_diffs = df["best_diff"].values
    n          = len(best_diffs)

    state           = 0   # 0=idle, 1=arm_raised
    frames_in_state = 0
    cooldown        = 0
    raise_start     = 0
    intervals       = []

    for i in range(n):
        if cooldown > 0:
            cooldown -= 1
            continue

        arm_up = best_diffs[i] < -RAISE_THRESHOLD

        if state == 0:
            if arm_up:
                state           = 1
                frames_in_state = 1
                raise_start     = i

        elif state == 1:
            if arm_up:
                frames_in_state += 1
                # Safety: if arm stays up too long, record and reset
                if frames_in_state > MAX_RAISED_FRAMES:
                    if frames_in_state >= MIN_RAISE_FRAMES:
                        intervals.append((raise_start, i))
                    state           = 0
                    frames_in_state = 0
                    cooldown        = COOLDOWN_FRAMES
            else:
                # Arm came back down — valid serve if raised long enough
                if frames_in_state >= MIN_RAISE_FRAMES:
                    intervals.append((raise_start, i))
                    cooldown = COOLDOWN_FRAMES
                state           = 0
                frames_in_state = 0

    return intervals


def interval_to_impact(df, raise_start, arm_down):
    """
    Within the serve interval [raise_start .. arm_down], return the
    positional index of the frame with the highest impact score.
    """
    window_scores = df["score"].iloc[raise_start : arm_down + 1]
    return window_scores.values.argmax() + raise_start


# ─────────────────────────────────────────────────────────────────────────────
# Per-serve CSV
# ─────────────────────────────────────────────────────────────────────────────

def extract_serve_window(df, impact_pos):
    """
    Return the rows spanning [impact_pos - PRE .. impact_pos + POST].
    Adds a relative_frame column (0 = first frame of the window).
    The best_diff column is dropped from the saved CSV (internal use only).
    """
    lo     = max(0, impact_pos - PRE_IMPACT_FRAMES)
    hi     = min(len(df), impact_pos + POST_IMPACT_FRAMES + 1)
    window = df.iloc[lo:hi].copy()
    window.insert(0, "relative_frame", range(len(window)))
    return window.drop(columns=["best_diff"])


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs", exist_ok=True)

    # Collect per-player max-angle lists:  { player: {metric: [values]} }
    all_maxes = {}

    for player, video_path in VIDEOS.items():
        print(f"\n{'='*55}")
        print(f"Player : {player}")
        print(f"Video  : {video_path}")

        if not os.path.exists(video_path):
            print("  [!] Video not found -- skipping.")
            continue

        # ── Output directory ─────────────────────────────────────────────────
        out_dir = f"outputs/{player}_serves"
        os.makedirs(out_dir, exist_ok=True)

        # ── Pose detection on all frames ─────────────────────────────────────
        print("  Running MediaPipe Pose on all frames...")
        df, fps, total_frames = process_video(video_path)
        print(f"  Pose detected : {len(df):,} / {total_frames:,} frames "
              f"({fps:.1f} fps, {total_frames/fps:.1f} s)")

        if df.empty:
            print("  [!] No pose data -- skipping.")
            continue

        # ── Wrist-based serve detection ──────────────────────────────────────
        intervals = find_serve_intervals(df)
        print(f"  Serves detected : {len(intervals)}")

        if not intervals:
            print("  [!] No serves found.")
            continue

        # ── Extract and save each serve ──────────────────────────────────────
        player_maxes = {"elbow": [], "knee": [], "shoulder": []}

        for serve_num, (rs, ad) in enumerate(intervals, start=1):
            impact_pos = interval_to_impact(df, rs, ad)
            serve_df   = extract_serve_window(df, impact_pos)

            csv_path = os.path.join(out_dir, f"serve_{serve_num}.csv")
            serve_df.to_csv(csv_path, index=False)

            e_max = serve_df["elbow_angle"].max()
            k_max = serve_df["knee_angle"].max()
            s_max = serve_df["shoulder_angle"].max()

            player_maxes["elbow"].append(e_max)
            player_maxes["knee"].append(k_max)
            player_maxes["shoulder"].append(s_max)

            impact_frame = int(df.iloc[impact_pos]["frame"])
            print(f"    Serve {serve_num:>2} | impact frame {impact_frame:>5} | "
                  f"elbow {e_max:>6.1f} | knee {k_max:>6.1f} | shoulder {s_max:>6.1f}")

        all_maxes[player] = player_maxes

    # ── Summary statistics ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Building summary_statistics.csv...")

    players_in_order = ["djokovic", "federer", "amateur"]
    metrics          = ["elbow", "knee", "shoulder"]

    rows = []
    for metric in metrics:
        row = {"Metric": f"{metric}_angle"}
        for player in players_in_order:
            vals = all_maxes.get(player, {}).get(metric, [])
            row[f"{player.capitalize()}_mean"] = round(float(np.mean(vals)),   2) if vals else None
            row[f"{player.capitalize()}_std"]  = round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 2) if vals else None
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = "outputs/summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved -> {summary_path}")

    # ── Box plots ────────────────────────────────────────────────────────────
    print("\nGenerating distribution_plots.png...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle("Serve Biomechanics -- Distribution of Peak Angles",
                 fontsize=14, fontweight="bold")

    player_labels = ["Djokovic", "Federer", "Amateur"]
    player_keys   = ["djokovic", "federer", "amateur"]
    metric_labels = ["Elbow Angle (deg)", "Knee Angle (deg)", "Shoulder Angle (deg)"]
    metric_keys   = ["elbow", "knee", "shoulder"]
    colors        = ["#2196F3", "#4CAF50", "#FF9800"]

    for ax, m_key, m_label in zip(axes, metric_keys, metric_labels):
        data, labels, plot_colors = [], [], []

        for p_key, p_label, color in zip(player_keys, player_labels, colors):
            vals = all_maxes.get(p_key, {}).get(m_key, [])
            if vals:
                data.append(vals)
                labels.append(p_label)
                plot_colors.append(color)

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})

        for patch, col in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        # Overlay individual points so even 2-serve players show their data
        for i, (vals, col) in enumerate(zip(data, plot_colors), start=1):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter([i + j for j in jitter], vals,
                       color=col, zorder=5, s=50,
                       edgecolors="white", linewidths=0.5)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(m_label, fontsize=11)
        ax.set_title(m_label, fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 200)

    plt.tight_layout()
    plot_path = "outputs/distribution_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
