# # src/model.py
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from config import TARGET_COLUMN, RANDOM_STATE
#
#
# def train_decision_tree(df):
#     X = df.drop(columns=[TARGET_COLUMN])
#     y = df[TARGET_COLUMN]
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
#     )
#
#     dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
#     dt_model.fit(X_train, y_train)
#
#     y_pred = dt_model.predict(X_test)
#     print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred))
#
#     return dt_model, X_test, y_test, y_pred
#
#
# def train_random_forest(df):
#     X = df.drop(columns=[TARGET_COLUMN])
#     y = df[TARGET_COLUMN]
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
#     )
#
#     rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
#     rf_model.fit(X_train, y_train)
#
#     y_pred = rf_model.predict(X_test)
#     print("Random Forest Classification Report:\n", classification_report(y_test, y_pred))
#
#     return rf_model, X_test, y_test, y_pred







from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import TARGET_COLUMN, RANDOM_STATE

def train_decision_tree(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=6
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred


def train_random_forest(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=10,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred
