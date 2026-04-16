from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.datasets import BinaryLabelDataset
except ImportError as exc:
    Reweighing = None
    BinaryLabelDataset = None
    AIF360_IMPORT_ERROR = exc
else:
    AIF360_IMPORT_ERROR = None


FEATURE_COLUMNS: List[str] = [
    "AGELAST",
    "SEX",
    "RACE",
    "MARRY15X",
    "EDUCYR",
    "FTSTU53X",
    "ACTDTY53",
    "HONRDC53",
    "RTHLTH53",
    "MNHLTH53",
    "HIBPDX",
    "CHDDX",
    "ANGIDX",
    "MIDX",
    "OHRTDX",
    "STRKDX",
    "EMPHDX",
    "CHBRON53",
    "CHOLDX",
    "CANCERDX",
    "DIABDX",
    "JTPAIN53",
    "ARTHDX",
    "ARTHTYPE",
    "ASTHDX",
    "ADHDADDX",
    "PREGNT53",
    "WLKLIM53",
    "ACTLIM53",
    "SOCLIM53",
    "COGLIM53",
    "DFHEAR42",
    "DFSEE42",
    "DFCOG42",
    "DFWLKC42",
    "DFDRSB42",
    "DFERND42",
    "INSCOV15",
    "POVCAT15",
    "REGION15",
]

RACE_INDEX = FEATURE_COLUMNS.index("RACE")
THRESH_GRID = np.linspace(0.01, 0.5, 50)

BINARY_FEATURES = {
    "RACE",
    "HIBPDX",
    "CHDDX",
    "ANGIDX",
    "MIDX",
    "OHRTDX",
    "STRKDX",
    "EMPHDX",
    "CHOLDX",
    "CANCERDX",
    "DIABDX",
    "ARTHDX",
    "ASTHDX",
    "PREGNT53",
    "WLKLIM53",
    "ACTLIM53",
    "SOCLIM53",
    "COGLIM53",
}

DEFAULT_INPUT: Dict[str, float] = {feature: 0.0 for feature in FEATURE_COLUMNS}
DEFAULT_INPUT.update(
    {
        "AGELAST": 45.0,
        "SEX": 1.0,
        "RACE": 1.0,
        "MARRY15X": 1.0,
        "EDUCYR": 12.0,
        "RTHLTH53": 2.0,
        "MNHLTH53": 2.0,
        "INSCOV15": 1.0,
        "POVCAT15": 3.0,
        "REGION15": 2.0,
    }
)

GROUPS: Dict[str, List[str]] = {
    "Demographics": [
        "AGELAST",
        "SEX",
        "RACE",
        "MARRY15X",
        "EDUCYR",
        "FTSTU53X",
        "ACTDTY53",
        "HONRDC53",
        "INSCOV15",
        "POVCAT15",
        "REGION15",
    ],
    "Health Status": ["RTHLTH53", "MNHLTH53"],
    "Diagnoses": [
        "HIBPDX",
        "CHDDX",
        "ANGIDX",
        "MIDX",
        "OHRTDX",
        "STRKDX",
        "EMPHDX",
        "CHBRON53",
        "CHOLDX",
        "CANCERDX",
        "DIABDX",
        "JTPAIN53",
        "ARTHDX",
        "ARTHTYPE",
        "ASTHDX",
        "ADHDADDX",
        "PREGNT53",
    ],
    "Functional Limitations": [
        "WLKLIM53",
        "ACTLIM53",
        "SOCLIM53",
        "COGLIM53",
        "DFHEAR42",
        "DFSEE42",
        "DFCOG42",
        "DFWLKC42",
        "DFDRSB42",
        "DFERND42",
    ],
}


def _resolve_data_path(explicit_path: str | None = None) -> Path:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = os.getenv("MEPS_CSV_PATH")
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "h181.csv",
            here.parent / "h181.csv",
            here.parent.parent / "h181.csv",
            Path.cwd() / "h181.csv",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find h181.csv. Checked:\n{joined}")


def _prepare_training_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    df["RACE"] = ((df["RACEV2X"] == 1) & (df["HISPANX"] == 2)).astype(int)

    util_cols = ["OBTOTV15", "OPTOTV15", "ERTOT15", "IPNGTD15", "HHTOTD15"]
    for col in util_cols:
        df[col] = df[col].clip(lower=0)

    df["UTILIZATION"] = df[util_cols].sum(axis=1)
    df["LABEL"] = (df["UTILIZATION"] >= 10).astype(int)

    for col in FEATURE_COLUMNS:
        if col != "RACE":
            df[col] = df[col].clip(lower=0)

    model_df = df[FEATURE_COLUMNS + ["LABEL", "PERWT15F"]].dropna().copy()
    return model_df


def _compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float((tpr + tnr) / 2.0)


def _compute_group_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    pos_rate = float(np.mean(y_pred)) if len(y_pred) > 0 else 0.0

    return {"tpr": tpr, "fpr": fpr, "pos_rate": pos_rate}


def _compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, race: np.ndarray) -> Dict[str, float]:
    unprivileged = race == 0
    privileged = race == 1

    rates_unpriv = _compute_group_rates(y_true[unprivileged], y_pred[unprivileged])
    rates_priv = _compute_group_rates(y_true[privileged], y_pred[privileged])

    disp_imp = (
        rates_unpriv["pos_rate"] / rates_priv["pos_rate"]
        if rates_priv["pos_rate"] > 0
        else 0.0
    )

    return {
        "balanced_accuracy": _compute_balanced_accuracy(y_true, y_pred),
        "disparate_impact": float(disp_imp),
        "statistical_parity_difference": float(rates_unpriv["pos_rate"] - rates_priv["pos_rate"]),
        "equal_opportunity_difference": float(rates_unpriv["tpr"] - rates_priv["tpr"]),
        "average_odds_difference": float(
            ((rates_unpriv["fpr"] - rates_priv["fpr"]) + (rates_unpriv["tpr"] - rates_priv["tpr"])) / 2.0
        ),
    }


def _best_threshold(model: Any, x_val: np.ndarray, y_val: np.ndarray) -> float:
    proba = model.predict_proba(x_val)[:, 1]
    scores = []
    for threshold in THRESH_GRID:
        pred = (proba >= threshold).astype(int)
        scores.append(_compute_balanced_accuracy(y_val, pred))

    best_idx = int(np.argmax(scores))
    return float(THRESH_GRID[best_idx])


def _create_aif_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    weight_data: np.ndarray,
) -> Any:
    df_temp = pd.DataFrame(x_data, columns=FEATURE_COLUMNS)
    df_temp["LABEL"] = y_data
    df_temp["instance_weights"] = weight_data

    return BinaryLabelDataset(
        df=df_temp,
        label_names=["LABEL"],
        protected_attribute_names=["RACE"],
        favorable_label=1.0,
        unfavorable_label=0.0,
        instance_weights_name="instance_weights",
    )


@dataclass
class ModelBundle:
    baseline_model: Any
    fair_model: Any
    baseline_threshold: float
    fair_threshold: float
    metrics: Dict[str, Any]


class MedicalBiasModelService:
    def __init__(self, artifact_path: str | None = None, data_path: str | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.artifact_path = Path(artifact_path) if artifact_path else base_dir / "models" / "medical_bias_models.joblib"
        self.data_path = _resolve_data_path(data_path)
        self.bundle: ModelBundle | None = None

    def load_or_train(self, force_retrain: bool = False) -> None:
        if self.bundle is not None and not force_retrain:
            return

        if AIF360_IMPORT_ERROR is not None:
            raise RuntimeError(
                "aif360 is required for debiased predictions. Install dependencies from requirements.txt."
            ) from AIF360_IMPORT_ERROR

        if self.artifact_path.exists() and not force_retrain:
            payload = joblib.load(self.artifact_path)
            self.bundle = ModelBundle(**payload)
            return

        model_df = _prepare_training_data(self.data_path)

        x = model_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = model_df["LABEL"].to_numpy(dtype=int)
        race = model_df["RACE"].to_numpy(dtype=int)
        weights = model_df["PERWT15F"].to_numpy(dtype=float)

        x_temp, x_test, y_temp, y_test, race_temp, race_test, w_temp, w_test = train_test_split(
            x,
            y,
            race,
            weights,
            test_size=0.2,
            random_state=1,
            stratify=y,
        )

        x_train, x_val, y_train, y_val, race_train, race_val, w_train, _ = train_test_split(
            x_temp,
            y_temp,
            race_temp,
            w_temp,
            test_size=0.375,
            random_state=1,
            stratify=y_temp,
        )

        baseline_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=1),
        )
        baseline_model.fit(x_train, y_train, logisticregression__sample_weight=w_train)
        baseline_threshold = _best_threshold(baseline_model, x_val, y_val)

        train_dataset = _create_aif_dataset(x_train, y_train, w_train)
        rw = Reweighing(
            unprivileged_groups=[{"RACE": 0}],
            privileged_groups=[{"RACE": 1}],
        )
        transformed_train = rw.fit_transform(train_dataset)

        fair_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=1),
        )
        fair_model.fit(
            transformed_train.features,
            transformed_train.labels.ravel(),
            logisticregression__sample_weight=transformed_train.instance_weights,
        )
        fair_threshold = _best_threshold(fair_model, x_val, y_val)

        baseline_test_pred = (
            baseline_model.predict_proba(x_test)[:, 1] >= baseline_threshold
        ).astype(int)
        fair_test_pred = (fair_model.predict_proba(x_test)[:, 1] >= fair_threshold).astype(int)

        metrics = {
            "dataset": {
                "rows": int(len(model_df)),
                "positive_rate": float(np.mean(y)),
                "group_positive_rate": {
                    "non_white": float(np.mean(y[race == 0])) if np.any(race == 0) else 0.0,
                    "white": float(np.mean(y[race == 1])) if np.any(race == 1) else 0.0,
                },
            },
            "without_bias_mitigation": _compute_fairness_metrics(y_test, baseline_test_pred, race_test),
            "with_bias_mitigation_reweighing": _compute_fairness_metrics(y_test, fair_test_pred, race_test),
            "thresholds": {
                "without_bias_mitigation": baseline_threshold,
                "with_bias_mitigation_reweighing": fair_threshold,
            },
        }

        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_model": baseline_model,
            "fair_model": fair_model,
            "baseline_threshold": baseline_threshold,
            "fair_threshold": fair_threshold,
            "metrics": metrics,
        }
        joblib.dump(payload, self.artifact_path)

        self.bundle = ModelBundle(**payload)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        self.load_or_train()
        assert self.bundle is not None

        x_values = []
        for feature in FEATURE_COLUMNS:
            raw_value = features.get(feature, DEFAULT_INPUT[feature])
            x_values.append(float(raw_value))

        x_row = np.array(x_values, dtype=float).reshape(1, -1)

        baseline_prob = float(self.bundle.baseline_model.predict_proba(x_row)[0, 1])
        fair_prob = float(self.bundle.fair_model.predict_proba(x_row)[0, 1])

        baseline_pred = int(baseline_prob >= self.bundle.baseline_threshold)
        fair_pred = int(fair_prob >= self.bundle.fair_threshold)

        return {
            "input": {feature: x_values[idx] for idx, feature in enumerate(FEATURE_COLUMNS)},
            "prediction_without_bias_mitigation": {
                "label": baseline_pred,
                "label_text": "High Utilization" if baseline_pred == 1 else "Low Utilization",
                "probability_high_utilization": baseline_prob,
                "threshold": self.bundle.baseline_threshold,
            },
            "prediction_with_bias_mitigation": {
                "label": fair_pred,
                "label_text": "High Utilization" if fair_pred == 1 else "Low Utilization",
                "probability_high_utilization": fair_prob,
                "threshold": self.bundle.fair_threshold,
            },
            "delta_probability_fair_minus_unfair": fair_prob - baseline_prob,
        }

    def get_metrics(self) -> Dict[str, Any]:
        self.load_or_train()
        assert self.bundle is not None
        return self.bundle.metrics

    def get_schema(self) -> Dict[str, Any]:
        grouped_features = []
        for group_name, features in GROUPS.items():
            entries = []
            for feature in features:
                if feature in BINARY_FEATURES:
                    entries.append(
                        {
                            "name": feature,
                            "min": 0,
                            "max": 1,
                            "step": 1,
                            "default": DEFAULT_INPUT[feature],
                        }
                    )
                elif feature == "AGELAST":
                    entries.append(
                        {
                            "name": feature,
                            "min": 0,
                            "max": 95,
                            "step": 1,
                            "default": DEFAULT_INPUT[feature],
                        }
                    )
                elif feature == "EDUCYR":
                    entries.append(
                        {
                            "name": feature,
                            "min": 0,
                            "max": 25,
                            "step": 1,
                            "default": DEFAULT_INPUT[feature],
                        }
                    )
                else:
                    entries.append(
                        {
                            "name": feature,
                            "min": 0,
                            "max": 10,
                            "step": 1,
                            "default": DEFAULT_INPUT[feature],
                        }
                    )
            grouped_features.append({"group": group_name, "features": entries})

        return {
            "feature_columns": FEATURE_COLUMNS,
            "groups": grouped_features,
            "notes": {
                "race": "RACE: 1 = White non-Hispanic (privileged), 0 = Non-White (unprivileged)",
                "label": "Prediction target is high healthcare utilization (>= 10 visits).",
            },
        }
