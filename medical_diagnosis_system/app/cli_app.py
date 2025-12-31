# # app/cli_app.py
# import sys
# sys.path.append("../src")
#
# import pandas as pd
# from utils import load_model
# from config import MODEL_PATH
# from preprocessing import clean_data
#
# def prompt_binary(name):
#     v = input(f"{name} (0/1 or Yes/No): ").strip()
#     if v.lower() in ('yes','y','1'): return 1
#     if v.lower() in ('no','n','0'): return 0
#     try:
#         return int(v)
#     except:
#         return 0
#
# def main():
#     model = load_model(MODEL_PATH)
#
#     print("Enter patient data. Leave blank for defaults.")
#     # minimal interactive; modify to match available features
#     age = input("Age: ").strip()
#     age = int(age) if age else 60
#
#     gender = prompt_binary("Gender (0=Female,1=Male)")
#
#     # For simplicity we ask a few key features; expand as desired:
#     bmi = input("BMI (float): ").strip()
#     bmi = float(bmi) if bmi else 25.0
#
#     mmse = input("MMSE score: ").strip()
#     mmse = float(mmse) if mmse else 24.0
#
#     # build a dict consistent with training features - for production, load feature_names_in_
#     sample = {
#         "Age": age,
#         "Gender": gender,
#         "BMI": bmi,
#         "MMSE": mmse
#     }
#
#     df = pd.DataFrame([sample])
#
#     # Align columns to model training set:
#     model_features = getattr(model, "feature_names_in_", None)
#     if model_features is not None:
#         for col in model_features:
#             if col not in df.columns:
#                 df[col] = 0
#         df = df[model_features]
#
#     pred = model.predict(df)
#     print("\nSuggested Diagnosis:", pred[0])
#
# if __name__ == "__main__":
#     main()
