from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, request

from model_service import MedicalBiasModelService


app = Flask(__name__)
service = MedicalBiasModelService()


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/schema")
def schema() -> Any:
    try:
        return jsonify(service.get_schema())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/metrics")
def metrics() -> Any:
    try:
        return jsonify(service.get_metrics())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/predict")
def predict() -> Any:
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    features = payload.get("features", payload)

    try:
        prediction = service.predict(features)
        return jsonify(prediction)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    service.load_or_train()
    app.run(host="0.0.0.0", port=int(os.getenv("FLASK_PORT", "5000")), debug=False)
