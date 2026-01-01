from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import TARGET_COLUMN, RANDOM_STATE
import joblib
import os


def train_random_forest(df, model_path):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
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

    print("=== Random Forest Results ===")
    print(classification_report(y_test, y_pred))

    # Save model + feature names together
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": feature_names
        },
        model_path
    )

    return model, X_test, y_test, y_pred
