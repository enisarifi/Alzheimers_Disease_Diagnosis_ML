import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "diagnosis_model.pkl")

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Alzheimer’s Diagnosis System",
    layout="centered"
)

st.title("🧠 Alzheimer’s Disease Diagnosis System")
st.write(
    "This tool provides a **decision-support risk estimate** based on patient data. "
    "It is **not** a medical diagnosis."
)

# ---------------- Load Model ----------------
model = joblib.load(MODEL_PATH)

# Get feature names used during training
feature_names = model.feature_names_in_

# ---------------- Helper Functions ----------------
def yes_no(label):
    return 1 if st.radio(label, ["No", "Yes"], horizontal=True) == "Yes" else 0

# ---------------- User Inputs ----------------
st.subheader("👤 Patient Information")

age = st.number_input("Age", min_value=40, max_value=100, value=65)
mmse = st.slider("MMSE Score", 0, 30, 20)

st.subheader("🧠 Cognitive & Behavioral Symptoms")

memory_loss = st.slider("Memory complaints severity", 1, 10, 5)
confusion = yes_no("Confusion episodes")
forgetfulness = yes_no("Frequent forgetfulness")
behavioral_issues = yes_no("Behavioral problems")

st.subheader("🏥 Daily Functioning")

adl = st.slider("Ability to perform daily activities (ADL)", 1, 10, 6)

# ---------------- Build Feature Vector ----------------
# Initialize all features to 0
input_dict = {feature: 0 for feature in feature_names}

# Fill only the features we actually collect
input_dict["Age"] = age
input_dict["MMSE"] = mmse
input_dict["MemoryComplaints"] = memory_loss
input_dict["Confusion"] = confusion
input_dict["Forgetfulness"] = forgetfulness
input_dict["BehavioralProblems"] = behavioral_issues
input_dict["ADL"] = adl

# Create DataFrame
input_data = pd.DataFrame([input_dict])

# ---------------- Prediction ----------------
if st.button("Predict Diagnosis"):
    proba = model.predict_proba(input_data)[0]
    alz_probability = proba[1] * 100  # Probability of Alzheimer’s

    st.subheader("📊 Result")

    if alz_probability < 30:
        st.success(f"✅ Low risk of Alzheimer’s ({alz_probability:.2f}%)")
    elif alz_probability < 60:
        st.warning(f"⚠️ Moderate risk of Alzheimer’s ({alz_probability:.2f}%)")
    else:
        st.error(f"🚨 High risk of Alzheimer’s ({alz_probability:.2f}%)")

    st.caption(
        "⚠️ This is a statistical risk estimate based on historical data. "
        "It should not be used as a standalone medical diagnosis."
    )
