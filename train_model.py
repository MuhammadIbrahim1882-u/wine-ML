import os
import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Order matches sklearn's load_wine() columns exactly.
FEATURES = [
    {"key": "alcohol", "label": "Alcohol (%)", "example": 13.0},
    {"key": "malic_acid", "label": "Malic Acid (g/L)", "example": 2.3},
    {"key": "ash", "label": "Ash (g/L)", "example": 2.4},
    {"key": "alcalinity_of_ash", "label": "Alcalinity of Ash", "example": 19.5},
    {"key": "magnesium", "label": "Magnesium (mg/L)", "example": 99},
    {"key": "total_phenols", "label": "Total Phenols", "example": 2.3},
    {"key": "flavanoids", "label": "Flavanoids", "example": 2.0},
    {"key": "nonflavanoid_phenols", "label": "Nonflavanoid Phenols", "example": 0.36},
    {"key": "proanthocyanins", "label": "Proanthocyanins", "example": 1.6},
    {"key": "color_intensity", "label": "Color Intensity", "example": 5.0},
    {"key": "hue", "label": "Hue", "example": 0.96},
    {"key": "od280", "label": "OD280/OD315 Ratio", "example": 2.6},
    {"key": "proline", "label": "Proline (mg/L)", "example": 746},
]

CLASS_NAMES = ["Cultivar 1", "Cultivar 2", "Cultivar 3"]


def train_and_save():
    data = load_wine()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cv_accuracy = cross_val_score(pipeline, X, y, cv=5).mean()

    importances = pipeline.named_steps["clf"].feature_importances_
    ranked = sorted(
        zip([f["label"] for f in FEATURES], importances),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    bundle = {
        "pipeline": pipeline,
        "accuracy": float(accuracy),
        "cv_accuracy": float(cv_accuracy),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "top_features": [
            {"label": name, "value": round(float(val) * 100, 1)} for name, val in ranked
        ],
        "data_max": [float(v) for v in X.max(axis=0)],
    }
    joblib.dump(bundle, MODEL_PATH)

    print("=" * 50)
    print("Model trained and saved to:", MODEL_PATH)
    print(f"Test accuracy        : {accuracy * 100:.2f}%")
    print(f"5-fold CV accuracy   : {cv_accuracy * 100:.2f}%")
    print("-" * 50)
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES))
    print("=" * 50)
    return bundle


if __name__ == "__main__":
    train_and_save()
