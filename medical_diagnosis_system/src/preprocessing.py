# src/preprocessing.py
import pandas as pd
from config import MIN_SAMPLES_PER_CLASS

DROP_COLUMNS = ["PatientID", "DoctorInCharge"]
# adjust if any of these don't exist in your CSV — we will handle gracefully

NUMERIC_COLUMNS = [
    "Age", "BMI", "SystolicBP", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
    "MMSE", "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems",
    "ADL", "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness"
]

# Binary-ish columns that may be 0/1 or small integers / ordinal categories
BINARY_COLUMNS = [
    "Gender", "Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
    "SleepQuality", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
    "Depression", "HeadInjury", "Hypertension"
]

def safe_drop_columns(df, cols):
    existing = [c for c in cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df

def group_rare_classes(df, target_col='Diagnosis', min_samples=MIN_SAMPLES_PER_CLASS):
    if target_col not in df.columns:
        return df
    counts = df[target_col].value_counts()
    rare = counts[counts < min_samples].index
    if len(rare) > 0:
        df[target_col] = df[target_col].replace(rare, 'Other')
    return df

def clean_data(df):
    # Basic cleaning
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    # Drop irrelevant columns if present
    df = safe_drop_columns(df, DROP_COLUMNS)

    # Attempt to coerce numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure binary columns are numeric
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Drop rows with missing target or too many NaNs
    if 'Diagnosis' in df.columns:
        df = df.dropna(subset=['Diagnosis'])

    # Optionally drop rows with many NaNs (here: drop if >30% missing)
    df = df[df.isnull().mean(axis=1) <= 0.3]

    # Drop remaining rows with critical NaNs
    df = df.dropna()

    # Group rare classes
    df = group_rare_classes(df, target_col='Diagnosis', min_samples=MIN_SAMPLES_PER_CLASS)

    # One-hot encode categorical fields that remain (Ethnicity, EducationLevel, maybe Gender if non-binary)
    categorical = []
    if 'Ethnicity' in df.columns:
        categorical.append('Ethnicity')
    if 'EducationLevel' in df.columns:
        categorical.append('EducationLevel')

    if categorical:
        df = pd.get_dummies(df, columns=categorical, drop_first=True)

    return df
