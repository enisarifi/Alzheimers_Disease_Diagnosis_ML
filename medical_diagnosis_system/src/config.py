# src/config.py
import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "alzheimers_disease_data.csv")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "alzheimers_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "diagnosis_model.pkl")

# Target column
TARGET_COLUMN = "Diagnosis"

# Random seed
RANDOM_STATE = 42

# Minimum samples per class; others grouped as "Other"
MIN_SAMPLES_PER_CLASS = 10
