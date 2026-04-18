# Tennis Serve Biomechanics Analysis

Computer vision project that extracts and compares body joint angles during the tennis serve motion across players of different skill levels.

**Stack:** Python · OpenCV · MediaPipe · Pandas · Matplotlib

---

## Project Goal

Use pose estimation to quantify the biomechanical differences between professional and amateur tennis serves — specifically: elbow extension, knee drive, shoulder rotation, and wrist velocity at the moment of ball contact.

Key question: **what separates a professional serve from an amateur's, in measurable terms?**

---

## Key Findings

| Metric | Djokovic | Federer | Amateur | Pro Advantage |
|--------|----------|---------|---------|---------------|
| Elbow angle at impact | ~180° | ~180° | ~157° | +23° extension |
| Knee angle at impact | 178.6° | 173.0° | 163.9° | +15° leg drive |
| Wrist velocity (max) | 0.0031 | — | 0.0109 | Pros: smoother, more controlled |

The most striking finding: the amateur achieves **23° less elbow extension** at impact, directly reducing power and consistency. Professionals generate force through the full kinetic chain (legs → trunk → shoulder → arm), not wrist snap.

---

## Project Structure

```
tennis_serve_ai_project/
├── src/
│   ├── analyze_serve.py     # Main pipeline: pose detection → angle extraction → CSV
│   ├── pose_detection.py    # Quick pose visualisation script
│   └── test_video.py        # Dependency check
├── notebooks/
│   └── o2_modeling.ipynb    # Full analysis: impact detection, comparison, conclusions
├── data/
│   └── raw_videos/          # Input videos (amateur.mp4, djokovic.mp4)
├── outputs/
│   ├── amateur_analysis.csv
│   ├── djokovic_analysis.csv
│   └── federer_analysis.csv
└── requirements.txt
```

---

## How It Works

### 1. Pose Detection
[MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) detects 33 body landmarks per frame from the raw video. The right-side joints (shoulder, elbow, wrist, hip, knee, ankle) are used for serve analysis.

### 2. Angle Extraction
For each frame where a pose is detected, three angles are computed using the law of cosines applied to landmark coordinates:

- **Elbow angle** — shoulder → elbow → wrist
- **Knee angle** — hip → knee → ankle
- **Shoulder angle** — hip → shoulder → elbow

### 3. Impact Frame Detection
The exact moment of ball contact is identified using a **biomechanics scoring function**:

```
score = elbow_angle + knee_angle − 300 × wrist_height
```

At impact, the arm is fully extended (maximum elbow angle), the body is rising (maximum knee extension), and the wrist is at its highest point. The frame with the maximum composite score is selected.

### 4. Comparative Analysis
The ±40-frame window around each player's impact is extracted, smoothed, and compared. Metrics include angular velocity, wrist velocity, and angle trajectories.

---

## Getting Started

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd tennis_serve_ai_project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Verify installation
python src/test_video.py

# 3. Run analysis on a video
python src/analyze_serve.py --video data/raw_videos/amateur.mp4

# 4. Open the analysis notebook
jupyter notebook notebooks/o2_modeling.ipynb
```

### CLI Options

```
python src/analyze_serve.py [options]

  --video    Path to input video  (default: data/raw_videos/amateur.mp4)
  --output   Path for output CSV  (default: outputs/amateur_analysis.csv)
  --start    First frame to process (default: 0)
  --end      Last frame to process  (default: 3000)
```

---

## Author

**Gonzalo Tano**
