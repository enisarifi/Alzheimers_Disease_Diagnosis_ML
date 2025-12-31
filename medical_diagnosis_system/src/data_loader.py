# src/data_loader.py
import pandas as pd
import os
from config import DATA_RAW_PATH

def load_raw_data(path=DATA_RAW_PATH):
    """Load raw CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)
