"""
test_video.py – Quick dependency check.

Run this script to verify that all required packages are installed correctly
before running the main analysis pipeline.
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("All dependencies OK.")
print(f"  OpenCV    : {cv2.__version__}")
print(f"  MediaPipe : {mp.__version__}")
print(f"  NumPy     : {np.__version__}")
print(f"  Pandas    : {pd.__version__}")
