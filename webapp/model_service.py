from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing
    from aif360.datasets import BinaryLabelDataset
except ImportError:
    DisparateImpactRemover = None
    Reweighing = None
    BinaryLabelDataset = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


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


def _compute_theil_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # AIF360-compatible benefit definition for binary classification.
    benefit = 1.0 + y_pred.astype(float) - y_true.astype(float)
    mean_benefit = float(np.mean(benefit))
    if mean_benefit <= 0:
        return 0.0

    ratio = benefit / mean_benefit
    valid = ratio > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(ratio[valid] * np.log(ratio[valid])))


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
        "disp_imp": float(disp_imp),
        "stat_par_diff": float(rates_unpriv["pos_rate"] - rates_priv["pos_rate"]),
        "eq_opp_diff": float(rates_unpriv["tpr"] - rates_priv["tpr"]),
        "avg_odds_diff": float(
            ((rates_unpriv["fpr"] - rates_priv["fpr"]) + (rates_unpriv["tpr"] - rates_priv["tpr"])) / 2.0
        ),
        "theil_ind": _compute_theil_index(y_true, y_pred),
    }


def _compute_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "bal_acc": _compute_balanced_accuracy(y_true, y_pred),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def _compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    race: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    perf = _compute_performance_metrics(y_true, y_pred, y_prob)
    fair = _compute_fairness_metrics(y_true, y_pred, race)
    return {**perf, **fair}


def _compute_metric_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    race: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, List[float]]:
    curve: Dict[str, List[float]] = {
        "bal_acc": [],
        "disp_imp": [],
        "avg_odds_diff": [],
        "stat_par_diff": [],
        "eq_opp_diff": [],
        "theil_ind": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }
    for threshold in thresholds:
        metrics = _compute_threshold_metrics(y_true, y_prob, race, float(threshold))
        for key in curve:
            curve[key].append(float(metrics[key]))
    return curve


def _best_threshold_from_curve(curve: Dict[str, List[float]], thresholds: np.ndarray) -> float:
    best_idx = int(np.argmax(np.asarray(curve["bal_acc"])))
    return float(thresholds[best_idx])


def _best_roc_like_threshold(curve: Dict[str, List[float]], thresholds: np.ndarray) -> float:
    disp = np.clip(np.asarray(curve["disp_imp"], dtype=float), 1e-6, None)
    di_error = 1.0 - np.minimum(disp, 1.0 / disp)
    score = np.asarray(curve["bal_acc"], dtype=float) - 0.5 * di_error - 0.25 * np.abs(
        np.asarray(curve["avg_odds_diff"], dtype=float)
    )
    best_idx = int(np.argmax(score))
    return float(thresholds[best_idx])


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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _local_logreg_train(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    lr: float,
    epochs: int,
) -> Tuple[np.ndarray, float]:
    w = coef.copy()
    b = float(intercept)
    n_samples = max(1, x_data.shape[0])

    for _ in range(epochs):
        probs = _sigmoid(x_data @ w + b)
        error = probs - y_data
        grad_w = (x_data.T @ error) / n_samples
        grad_b = float(np.mean(error))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def _make_client_splits(n_samples: int, n_clients: int, rng: np.random.Generator) -> List[np.ndarray]:
    shuffled = rng.permutation(n_samples)
    return [arr for arr in np.array_split(shuffled, n_clients) if len(arr) > 0]


def _train_fedavg_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_clients: int,
    rounds: int,
    lr: float,
    local_epochs: int,
    dp_clip_norm: float | None = None,
    dp_noise_std: float = 0.0,
    seed: int = 1,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n_features = x_train.shape[1]
    coef = np.zeros(n_features, dtype=float)
    intercept = 0.0

    client_splits = _make_client_splits(x_train.shape[0], n_clients, rng)
    if not client_splits:
        return coef, intercept

    for _ in range(rounds):
        delta_w_accum = np.zeros_like(coef)
        delta_b_accum = 0.0
        weight_accum = 0.0

        for split in client_splits:
            x_client = x_train[split]
            y_client = y_train[split]

            w_local, b_local = _local_logreg_train(
                x_client,
                y_client,
                coef,
                intercept,
                lr=lr,
                epochs=local_epochs,
            )

            delta_w = w_local - coef
            delta_b = b_local - intercept

            if dp_clip_norm is not None and dp_clip_norm > 0:
                norm = float(np.sqrt(np.sum(delta_w**2) + delta_b**2))
                if norm > dp_clip_norm:
                    scale = dp_clip_norm / (norm + 1e-12)
                    delta_w *= scale
                    delta_b *= scale

            client_weight = float(len(split))
            delta_w_accum += client_weight * delta_w
            delta_b_accum += client_weight * delta_b
            weight_accum += client_weight

        if weight_accum > 0:
            avg_delta_w = delta_w_accum / weight_accum
            avg_delta_b = delta_b_accum / weight_accum
        else:
            avg_delta_w = np.zeros_like(coef)
            avg_delta_b = 0.0

        if dp_noise_std > 0 and dp_clip_norm is not None and dp_clip_norm > 0:
            avg_delta_w += rng.normal(0.0, dp_noise_std * dp_clip_norm, size=avg_delta_w.shape)
            avg_delta_b += float(rng.normal(0.0, dp_noise_std * dp_clip_norm))

        coef += avg_delta_w
        intercept += avg_delta_b

    return coef, float(intercept)


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


@dataclass
class ModelBundle:
    sklearn_models: Dict[str, Any]
    linear_models: Dict[str, Dict[str, Any]]
    model_sources: Dict[str, Dict[str, str]]
    model_thresholds: Dict[str, float]
    model_classifiers: Dict[str, str]
    preferred_baseline: str
    preferred_fair: str
    metrics: Dict[str, Any]


class MedicalBiasModelService:
    def __init__(self, artifact_path: str | None = None, data_path: str | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.artifact_path = Path(artifact_path) if artifact_path else base_dir / "models" / "medical_bias_models.joblib"
        self.data_path = _resolve_data_path(data_path)
        self.bundle: ModelBundle | None = None

    def _predict_proba_for_model(self, model_name: str, x_row: np.ndarray) -> float:
        assert self.bundle is not None

        source = self.bundle.model_sources.get(model_name, {})
        kind = source.get("kind", "")
        ref = source.get("ref", model_name)

        if kind == "sklearn":
            model = self.bundle.sklearn_models[ref]
            return float(model.predict_proba(x_row)[0, 1])

        if kind == "linear":
            cfg = self.bundle.linear_models[ref]
            mean = np.asarray(cfg["mean"], dtype=float)
            scale = np.asarray(cfg["scale"], dtype=float)
            coef = np.asarray(cfg["coef"], dtype=float)
            intercept = float(cfg["intercept"])
            x_scaled = (x_row - mean.reshape(1, -1)) / scale.reshape(1, -1)
            return float(_sigmoid(x_scaled @ coef + intercept).ravel()[0])

        raise KeyError(f"Unknown model source for {model_name}")

    def load_or_train(self, force_retrain: bool = False) -> None:
        if self.bundle is not None and not force_retrain:
            return

        if self.artifact_path.exists() and not force_retrain:
            payload = joblib.load(self.artifact_path)
            try:
                self.bundle = ModelBundle(**payload)
                return
            except Exception:
                # Artifact shape changed, re-train with the latest schema.
                force_retrain = True

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

        sklearn_models: Dict[str, Any] = {}
        linear_models: Dict[str, Dict[str, Any]] = {}
        model_sources: Dict[str, Dict[str, str]] = {}
        model_thresholds: Dict[str, float] = {}
        model_classifiers: Dict[str, str] = {}
        model_privacy: Dict[str, float] = {}
        model_explainability: Dict[str, float] = {}
        model_records: Dict[str, Dict[str, Any]] = {}

        def register_sklearn_model(name: str, model: Any, classifier: str) -> None:
            sklearn_models[name] = model
            model_sources[name] = {"kind": "sklearn", "ref": name}
            model_classifiers[name] = classifier

        def register_linear_model(name: str, cfg: Dict[str, Any], classifier: str) -> None:
            linear_models[name] = cfg
            model_sources[name] = {"kind": "linear", "ref": name}
            model_classifiers[name] = classifier

        baseline_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=1),
        )
        baseline_lr.fit(x_train, y_train, logisticregression__sample_weight=w_train)
        register_sklearn_model("Logistic Regression (Original)", baseline_lr, "Logistic Regression")
        model_privacy["Logistic Regression (Original)"] = 0.0
        model_explainability["Logistic Regression (Original)"] = 0.95

        baseline_rf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=25, random_state=1, n_jobs=-1),
        )
        baseline_rf.fit(x_train, y_train, randomforestclassifier__sample_weight=w_train)
        register_sklearn_model("Random Forest (Original)", baseline_rf, "Random Forest")
        model_privacy["Random Forest (Original)"] = 0.0
        model_explainability["Random Forest (Original)"] = 0.55

        if Reweighing is not None and BinaryLabelDataset is not None:
            train_dataset = _create_aif_dataset(x_train, y_train, w_train)
            rw = Reweighing(
                unprivileged_groups=[{"RACE": 0}],
                privileged_groups=[{"RACE": 1}],
            )
            transformed_train = rw.fit_transform(train_dataset)
            lr_rw = make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear", random_state=1),
            )
            lr_rw.fit(
                transformed_train.features,
                transformed_train.labels.ravel(),
                logisticregression__sample_weight=transformed_train.instance_weights,
            )
            register_sklearn_model("Logistic Regression (Reweighing)", lr_rw, "Logistic Regression")
            model_privacy["Logistic Regression (Reweighing)"] = 0.0
            model_explainability["Logistic Regression (Reweighing)"] = 0.95

        if DisparateImpactRemover is not None:
            try:
                dir_model = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="RACE")
                x_train_df = pd.DataFrame(x_train, columns=FEATURE_COLUMNS)
                x_train_dir = dir_model.fit_transform(x_train_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
                lr_dir = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(solver="liblinear", random_state=1),
                )
                lr_dir.fit(x_train_dir, y_train, logisticregression__sample_weight=w_train)

                class _DirWrappedModel:
                    def __init__(self, inner: Any, transformer: Any) -> None:
                        self.inner = inner
                        self.transformer = transformer

                    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
                        x_df = pd.DataFrame(x_data, columns=FEATURE_COLUMNS)
                        x_trans = self.transformer.transform(x_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
                        return self.inner.predict_proba(x_trans)

                register_sklearn_model(
                    "Disparate Impact Remover (LR)",
                    _DirWrappedModel(lr_dir, dir_model),
                    "Pre-processing + Logistic Regression",
                )
                model_privacy["Disparate Impact Remover (LR)"] = 0.0
                model_explainability["Disparate Impact Remover (LR)"] = 0.9
            except Exception:
                # AIF360's DIR may require optional BlackBoxAuditing dependency.
                # If unavailable, continue with the rest of the training pipeline.
                pass

        if SMOTE is not None:
            try:
                smote = SMOTE(random_state=1)
                x_smote, y_smote = smote.fit_resample(x_train, y_train)
                lr_smote = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(solver="liblinear", random_state=1),
                )
                lr_smote.fit(x_smote, y_smote)
                register_sklearn_model("SMOTE Oversampling (LR)", lr_smote, "Class Imbalance + Logistic Regression")
                model_privacy["SMOTE Oversampling (LR)"] = 0.0
                model_explainability["SMOTE Oversampling (LR)"] = 0.9
            except Exception:
                pass

        train_mean = np.mean(x_train, axis=0)
        train_scale = np.std(x_train, axis=0)
        train_scale = np.where(train_scale == 0, 1.0, train_scale)
        x_train_std = (x_train - train_mean) / train_scale
        x_val_std = (x_val - train_mean) / train_scale
        x_test_std = (x_test - train_mean) / train_scale

        fed_coef, fed_intercept = _train_fedavg_logreg(
            x_train_std,
            y_train.astype(float),
            n_clients=8,
            rounds=20,
            lr=0.08,
            local_epochs=1,
            seed=1,
        )
        register_linear_model(
            "Federated Learning (FedAvg)",
            {
                "coef": fed_coef,
                "intercept": fed_intercept,
                "mean": train_mean,
                "scale": train_scale,
            },
            "Federated Logistic Regression",
        )
        model_privacy["Federated Learning (FedAvg)"] = 0.65
        model_explainability["Federated Learning (FedAvg)"] = 0.90

        dp_coef, dp_intercept = _train_fedavg_logreg(
            x_train_std,
            y_train.astype(float),
            n_clients=8,
            rounds=20,
            lr=0.08,
            local_epochs=1,
            dp_clip_norm=1.0,
            dp_noise_std=0.05,
            seed=7,
        )
        register_linear_model(
            "DP Federated Learning (DP-FedAvg)",
            {
                "coef": dp_coef,
                "intercept": dp_intercept,
                "mean": train_mean,
                "scale": train_scale,
            },
            "DP Federated Logistic Regression",
        )
        model_privacy["DP Federated Learning (DP-FedAvg)"] = 1.0
        model_explainability["DP Federated Learning (DP-FedAvg)"] = 0.85

        y_val_probs: Dict[str, np.ndarray] = {}
        y_test_probs: Dict[str, np.ndarray] = {}

        for name in list(model_sources):
            if model_sources[name]["kind"] == "sklearn":
                y_val_probs[name] = np.asarray(sklearn_models[model_sources[name]["ref"]].predict_proba(x_val)[:, 1], dtype=float)
                y_test_probs[name] = np.asarray(sklearn_models[model_sources[name]["ref"]].predict_proba(x_test)[:, 1], dtype=float)
            elif model_sources[name]["kind"] == "linear":
                cfg = linear_models[model_sources[name]["ref"]]
                coef = np.asarray(cfg["coef"], dtype=float)
                intercept = float(cfg["intercept"])
                y_val_probs[name] = _sigmoid(x_val_std @ coef + intercept)
                y_test_probs[name] = _sigmoid(x_test_std @ coef + intercept)

        for name in list(y_val_probs):
            curve = _compute_metric_curve(y_val, y_val_probs[name], race_val, THRESH_GRID)
            threshold = _best_threshold_from_curve(curve, THRESH_GRID)
            model_thresholds[name] = threshold
            record = _compute_threshold_metrics(y_test, y_test_probs[name], race_test, threshold)
            record.update(
                {
                    "Model": name,
                    "Classifier": model_classifiers.get(name, "Model"),
                    "threshold": float(threshold),
                    "privacy_score": float(model_privacy.get(name, 0.0)),
                    "explainability_score": float(model_explainability.get(name, 0.5)),
                }
            )
            model_records[name] = record

        if "Logistic Regression (Original)" in y_val_probs:
            roc_curve = _compute_metric_curve(
                y_val,
                y_val_probs["Logistic Regression (Original)"],
                race_val,
                THRESH_GRID,
            )
            roc_threshold = _best_roc_like_threshold(roc_curve, THRESH_GRID)
            model_thresholds["Reject Option Classification (ROC-inspired)"] = roc_threshold
            model_sources["Reject Option Classification (ROC-inspired)"] = {
                "kind": "sklearn",
                "ref": "Logistic Regression (Original)",
            }
            model_classifiers["Reject Option Classification (ROC-inspired)"] = "Post-processing"
            model_privacy["Reject Option Classification (ROC-inspired)"] = 0.0
            model_explainability["Reject Option Classification (ROC-inspired)"] = 0.7
            roc_record = _compute_threshold_metrics(
                y_test,
                y_test_probs["Logistic Regression (Original)"],
                race_test,
                roc_threshold,
            )
            roc_record.update(
                {
                    "Model": "Reject Option Classification (ROC-inspired)",
                    "Classifier": "Post-processing",
                    "threshold": float(roc_threshold),
                    "privacy_score": 0.0,
                    "explainability_score": 0.7,
                }
            )
            model_records["Reject Option Classification (ROC-inspired)"] = roc_record

        comparison_df = pd.DataFrame(model_records.values())

        di = np.clip(comparison_df["disp_imp"].to_numpy(dtype=float), 1e-6, None)
        comparison_df["di_error"] = 1.0 - np.minimum(di, 1.0 / di)
        comparison_df["abs_avg_odds_diff"] = np.abs(comparison_df["avg_odds_diff"].to_numpy(dtype=float))
        comparison_df["abs_stat_par_diff"] = np.abs(comparison_df["stat_par_diff"].to_numpy(dtype=float))
        comparison_df["abs_eq_opp_diff"] = np.abs(comparison_df["eq_opp_diff"].to_numpy(dtype=float))
        comparison_df["bias_index"] = comparison_df[
            ["di_error", "abs_avg_odds_diff", "abs_stat_par_diff", "abs_eq_opp_diff"]
        ].mean(axis=1)
        bias_min = float(comparison_df["bias_index"].min())
        bias_span = float(comparison_df["bias_index"].max() - bias_min)
        comparison_df["fairness_score"] = 1.0 - (comparison_df["bias_index"] - bias_min) / (bias_span + 1e-12)
        comparison_df["fairness_score"] = comparison_df["fairness_score"].clip(0.0, 1.0)
        comparison_df["acc_fair_tradeoff"] = 0.5 * comparison_df["bal_acc"] + 0.5 * comparison_df["fairness_score"]
        comparison_df["acc_privacy_tradeoff"] = 0.7 * comparison_df["bal_acc"] + 0.3 * comparison_df["privacy_score"]
        comparison_df["acc_explain_tradeoff"] = 0.7 * comparison_df["bal_acc"] + 0.3 * comparison_df[
            "explainability_score"
        ]

        preferred_baseline = "Logistic Regression (Original)"
        fair_candidates = [
            "Logistic Regression (Reweighing)",
            "Disparate Impact Remover (LR)",
            "SMOTE Oversampling (LR)",
        ]
        preferred_fair = next((name for name in fair_candidates if name in model_records), preferred_baseline)

        baseline_probs_test = y_test_probs.get(preferred_baseline, np.zeros_like(y_test, dtype=float))
        residuals = y_test.astype(float) - baseline_probs_test.astype(float)
        residual_nonwhite = residuals[race_test == 0]
        residual_white = residuals[race_test == 1]

        def _group_residual_stats(values: np.ndarray) -> Dict[str, float | None]:
            if values.size == 0:
                return {"median_cdf_at_0": None, "lower_q_alpha_0_1": None, "upper_q_alpha_0_1": None, "mean": None}
            return {
                "median_cdf_at_0": _to_float(np.mean(values <= 0)),
                "lower_q_alpha_0_1": _to_float(np.quantile(values, 0.05)),
                "upper_q_alpha_0_1": _to_float(np.quantile(values, 0.95)),
                "mean": _to_float(np.mean(values)),
            }

        ks_stat = None
        ks_pvalue = None
        if stats is not None and residual_nonwhite.size > 1 and residual_white.size > 1:
            ks_res = stats.ks_2samp(residual_nonwhite, residual_white)
            ks_stat = _to_float(ks_res.statistic)
            ks_pvalue = _to_float(ks_res.pvalue)

        highlights = {}
        if not comparison_df.empty:
            best_acc_row = comparison_df.loc[comparison_df["bal_acc"].idxmax()]
            best_fair_row = comparison_df.loc[comparison_df["bias_index"].idxmin()]
            best_acc_fair_row = comparison_df.loc[comparison_df["acc_fair_tradeoff"].idxmax()]
            highlights = {
                "highest_accuracy": {
                    "model": str(best_acc_row["Model"]),
                    "bal_acc": _to_float(best_acc_row["bal_acc"]),
                },
                "lowest_bias": {
                    "model": str(best_fair_row["Model"]),
                    "bias_index": _to_float(best_fair_row["bias_index"]),
                },
                "best_accuracy_fairness_balance": {
                    "model": str(best_acc_fair_row["Model"]),
                    "score": _to_float(best_acc_fair_row["acc_fair_tradeoff"]),
                },
            }

        def _legacy_metrics(record: Dict[str, Any]) -> Dict[str, float | None]:
            return {
                "balanced_accuracy": _to_float(record.get("bal_acc")),
                "disparate_impact": _to_float(record.get("disp_imp")),
                "statistical_parity_difference": _to_float(record.get("stat_par_diff")),
                "equal_opportunity_difference": _to_float(record.get("eq_opp_diff")),
                "average_odds_difference": _to_float(record.get("avg_odds_diff")),
                "theil_index": _to_float(record.get("theil_ind")),
                "precision": _to_float(record.get("precision")),
                "recall": _to_float(record.get("recall")),
                "f1": _to_float(record.get("f1")),
                "roc_auc": _to_float(record.get("roc_auc")),
            }

        baseline_record = model_records[preferred_baseline]
        fair_record = model_records[preferred_fair]

        metrics = {
            "dataset": {
                "rows": int(len(model_df)),
                "positive_rate": float(np.mean(y)),
                "group_positive_rate": {
                    "non_white": float(np.mean(y[race == 0])) if np.any(race == 0) else 0.0,
                    "white": float(np.mean(y[race == 1])) if np.any(race == 1) else 0.0,
                },
            },
            "without_bias_mitigation": _legacy_metrics(baseline_record),
            "with_bias_mitigation_reweighing": _legacy_metrics(fair_record),
            "thresholds": {
                "without_bias_mitigation": float(model_thresholds.get(preferred_baseline, 0.5)),
                "with_bias_mitigation_reweighing": float(model_thresholds.get(preferred_fair, 0.5)),
            },
            "model_comparison": [
                {key: (_to_float(value) if isinstance(value, (float, int, np.floating, np.integer)) else value)
                 for key, value in row.items()}
                for row in comparison_df.to_dict(orient="records")
            ],
            "tradeoff_highlights": highlights,
            "signed_residual_analysis": {
                "baseline_model": preferred_baseline,
                "overall_mean": _to_float(np.mean(residuals)),
                "non_white": _group_residual_stats(residual_nonwhite),
                "white": _group_residual_stats(residual_white),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
            },
            "available_methods": sorted(list(model_records.keys())),
            "preferred_models": {
                "baseline": preferred_baseline,
                "fair": preferred_fair,
            },
        }

        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sklearn_models": sklearn_models,
            "linear_models": linear_models,
            "model_sources": model_sources,
            "model_thresholds": model_thresholds,
            "model_classifiers": model_classifiers,
            "preferred_baseline": preferred_baseline,
            "preferred_fair": preferred_fair,
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

        predictions_by_model: List[Dict[str, Any]] = []
        for model_name in sorted(self.bundle.model_thresholds.keys()):
            try:
                prob = self._predict_proba_for_model(model_name, x_row)
            except Exception:
                continue
            threshold = float(self.bundle.model_thresholds.get(model_name, 0.5))
            label = int(prob >= threshold)
            predictions_by_model.append(
                {
                    "model": model_name,
                    "classifier": self.bundle.model_classifiers.get(model_name, "Model"),
                    "label": label,
                    "label_text": "High Utilization" if label == 1 else "Low Utilization",
                    "probability_high_utilization": prob,
                    "threshold": threshold,
                }
            )

        baseline_name = self.bundle.preferred_baseline
        fair_name = self.bundle.preferred_fair

        baseline_obj = next((item for item in predictions_by_model if item["model"] == baseline_name), None)
        fair_obj = next((item for item in predictions_by_model if item["model"] == fair_name), None)

        if baseline_obj is None and predictions_by_model:
            baseline_obj = predictions_by_model[0]
        if fair_obj is None and predictions_by_model:
            fair_obj = predictions_by_model[min(1, len(predictions_by_model) - 1)]

        baseline_prob = float(baseline_obj["probability_high_utilization"]) if baseline_obj else 0.0
        fair_prob = float(fair_obj["probability_high_utilization"]) if fair_obj else 0.0
        baseline_label = int(baseline_obj["label"]) if baseline_obj else 0
        fair_label = int(fair_obj["label"]) if fair_obj else 0
        baseline_threshold = float(baseline_obj["threshold"]) if baseline_obj else 0.5
        fair_threshold = float(fair_obj["threshold"]) if fair_obj else 0.5

        return {
            "input": {feature: x_values[idx] for idx, feature in enumerate(FEATURE_COLUMNS)},
            "prediction_without_bias_mitigation": {
                "model": baseline_name,
                "label": baseline_label,
                "label_text": "High Utilization" if baseline_label == 1 else "Low Utilization",
                "probability_high_utilization": baseline_prob,
                "threshold": baseline_threshold,
            },
            "prediction_with_bias_mitigation": {
                "model": fair_name,
                "label": fair_label,
                "label_text": "High Utilization" if fair_label == 1 else "Low Utilization",
                "probability_high_utilization": fair_prob,
                "threshold": fair_threshold,
            },
            "delta_probability_fair_minus_unfair": fair_prob - baseline_prob,
            "predictions_by_model": predictions_by_model,
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
