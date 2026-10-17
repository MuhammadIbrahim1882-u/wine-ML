import math
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from train_model import CLASS_NAMES, FEATURES, MODEL_PATH, train_and_save

app = Flask(__name__)

# Train automatically if model.pkl is missing.
if not os.path.exists(MODEL_PATH):
    print("model.pkl not found - training a new model...")
    train_and_save()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]


def parse_inputs(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    values = []
    for i, feature in enumerate(FEATURES):
        raw = payload.get(feature["key"])
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"'{feature['label']}' is required.")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{feature['label']}' must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"'{feature['label']}' must be a valid number.")
        upper = bundle["data_max"][i] * 3
        if value < 0 or value > upper:
            raise ValueError(
                f"'{feature['label']}' must be between 0 and {upper:.1f}."
            )
        values.append(value)
    return values


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES, m=bundle)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        values = parse_inputs(payload)
        X = np.array([values])
        probabilities = pipeline.predict_proba(X)[0]
        best = int(np.argmax(probabilities))
        return jsonify(
            {
                "prediction": CLASS_NAMES[best],
                "confidence": round(float(probabilities[best]) * 100, 1),
                "probabilities": [
                    {"label": CLASS_NAMES[i], "value": round(float(p) * 100, 1)}
                    for i, p in enumerate(probabilities)
                ],
            }
        )
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception:
        return jsonify({"error": "Something went wrong on the server."}), 500


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Page not found."}), 404


if __name__ == "__main__":
    app.run(debug=True)
