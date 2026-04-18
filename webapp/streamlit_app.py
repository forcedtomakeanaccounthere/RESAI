from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

from model_service import MedicalBiasModelService


st.set_page_config(
    page_title="Healthcare Bias Demo",
    page_icon="stethoscope",
    layout="wide",
)

DEFAULT_API_URL = os.getenv("FLASK_API_URL", "http://127.0.0.1:5000")


@st.cache_resource
def get_local_service() -> MedicalBiasModelService:
    service = MedicalBiasModelService()
    service.load_or_train()
    return service


def call_api(path: str, method: str = "GET", payload: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    url = f"{DEFAULT_API_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=15)
        else:
            response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_schema(source_mode: str) -> Dict[str, Any]:
    if source_mode == "Flask API":
        schema = call_api("/schema")
        if schema is not None:
            return schema

    return get_local_service().get_schema()


def get_metrics(source_mode: str) -> Dict[str, Any]:
    if source_mode == "Flask API":
        metrics = call_api("/metrics")
        if metrics is not None:
            return metrics

    return get_local_service().get_metrics()


def predict(features: Dict[str, float], source_mode: str) -> Dict[str, Any]:
    if source_mode == "Flask API":
        result = call_api("/predict", method="POST", payload={"features": features})
        if result is not None:
            return result

    return get_local_service().predict(features)


def render_feature_input(name: str, spec: Dict[str, Any]) -> float:
    default_value = int(spec["default"])

    if name == "RACE":
        race_label = st.selectbox(
            "RACE",
            options=["Non-White", "White non-Hispanic"],
            index=1 if default_value == 1 else 0,
            help="1 = White non-Hispanic (privileged), 0 = Non-White (unprivileged)",
        )
        return float(1 if race_label == "White non-Hispanic" else 0)

    if name == "SEX":
        sex_label = st.selectbox(
            "SEX",
            options=["1 = Male", "2 = Female"],
            index=0 if default_value == 1 else 1,
        )
        return float(1 if sex_label.startswith("1") else 2)

    return float(
        st.number_input(
            name,
            min_value=int(spec["min"]),
            max_value=int(spec["max"]),
            value=default_value,
            step=int(spec["step"]),
        )
    )


def _format_metric_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    comparison = metrics.get("model_comparison")
    if comparison:
        df = pd.DataFrame(comparison)
        preferred_cols = [
            "Model",
            "Classifier",
            "bal_acc",
            "disp_imp",
            "avg_odds_diff",
            "stat_par_diff",
            "eq_opp_diff",
            "theil_ind",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "fairness_score",
            "privacy_score",
            "explainability_score",
            "bias_index",
            "threshold",
        ]
        keep = [col for col in preferred_cols if col in df.columns]
        if keep:
            return df[keep]
        return df

    # Backward compatibility with older metric payload.
    fallback_rows = [
        {
            "Model": "Without Bias Mitigation",
            **metrics.get("without_bias_mitigation", {}),
        },
        {
            "Model": "With Bias Mitigation (Reweighing)",
            **metrics.get("with_bias_mitigation_reweighing", {}),
        },
    ]
    return pd.DataFrame(fallback_rows)


st.title("Healthcare Utilization Bias Explorer")
st.caption(
    "Predict high healthcare utilization using the same MEPS HC-181 feature set and compare outputs with and without bias mitigation."
)

with st.sidebar:
    st.header("Runtime Options")
    source_mode = st.radio(
        "Prediction source",
        options=["Flask API", "Local in Streamlit"],
        index=0,
        help="Use Flask API mode when flask_api.py is running. Local mode directly loads the same model service in Streamlit.",
    )

    if source_mode == "Flask API":
        api_health = call_api("/health")
        if api_health is None:
            st.warning("Flask API is not reachable. The app will automatically fall back to local mode.")
        else:
            st.success("Flask API is running.")

schema = get_schema(source_mode)
metrics = get_metrics(source_mode)

with st.expander("How to interpret this demo", expanded=True):
    st.markdown(
        """
        - Outcome: High Utilization means expected healthcare visits >= 10.
        - Without bias mitigation: Logistic Regression trained on original weighted data.
        - With bias mitigation: Logistic Regression trained after AIF360 Reweighing.
        - Sensitive attribute: RACE (1 = White non-Hispanic, 0 = Non-White).
        """
    )

st.subheader("Enter Customer Details")
feature_values: Dict[str, float] = {}

with st.form("prediction_form"):
    for group in schema["groups"]:
        st.markdown(f"### {group['group']}")
        columns = st.columns(3)
        for idx, feature_spec in enumerate(group["features"]):
            feature_name = feature_spec["name"]
            with columns[idx % 3]:
                feature_values[feature_name] = render_feature_input(feature_name, feature_spec)

    submit = st.form_submit_button("Predict With and Without Bias Mitigation")

if submit:
    output = predict(feature_values, source_mode)
    st.session_state["prediction_output"] = output

if "prediction_output" in st.session_state:
    output = st.session_state["prediction_output"]

    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)

    unfair = output["prediction_without_bias_mitigation"]
    fair = output["prediction_with_bias_mitigation"]

    with col1:
        st.markdown("#### Without Bias Mitigation")
        st.metric("Predicted Label", unfair["label_text"])
        st.metric("P(High Utilization)", f"{unfair['probability_high_utilization']:.3f}")
        st.progress(min(max(unfair["probability_high_utilization"], 0.0), 1.0))

    with col2:
        st.markdown("#### With Bias Mitigation")
        st.metric("Predicted Label", fair["label_text"])
        st.metric("P(High Utilization)", f"{fair['probability_high_utilization']:.3f}")
        st.progress(min(max(fair["probability_high_utilization"], 0.0), 1.0))

    delta = output["delta_probability_fair_minus_unfair"]
    st.info(f"Probability Shift (fair - unfair): {delta:+.3f}")

    all_predictions = output.get("predictions_by_model", [])
    if all_predictions:
        st.markdown("#### All Model Predictions")
        pred_df = pd.DataFrame(all_predictions)
        st.dataframe(pred_df, use_container_width=True)

st.subheader("Model Fairness Snapshot")
comparison_df = _format_metric_table(metrics)
st.dataframe(comparison_df, use_container_width=True)

thresholds = metrics.get("thresholds", {})
st.caption(
    " | ".join(
        [
            f"Threshold (without mitigation): {thresholds.get('without_bias_mitigation', 0.5):.3f}",
            f"Threshold (with mitigation): {thresholds.get('with_bias_mitigation_reweighing', 0.5):.3f}",
        ]
    )
)

highlights = metrics.get("tradeoff_highlights", {})
if highlights:
    st.subheader("Trade-off Highlights")
    highest_acc = highlights.get("highest_accuracy", {})
    lowest_bias = highlights.get("lowest_bias", {})
    best_balance = highlights.get("best_accuracy_fairness_balance", {})

    st.markdown(
        "\n".join(
            [
                f"- Highest accuracy: **{highest_acc.get('model', 'N/A')}** (bal_acc={highest_acc.get('bal_acc', 0):.3f})",
                f"- Lowest measured bias: **{lowest_bias.get('model', 'N/A')}** (bias_index={lowest_bias.get('bias_index', 0):.3f})",
                f"- Best accuracy-fairness balance: **{best_balance.get('model', 'N/A')}** (score={best_balance.get('score', 0):.3f})",
            ]
        )
    )

signed_residual = metrics.get("signed_residual_analysis", {})
if signed_residual:
    st.subheader("Signed Residual Bias Check")
    st.caption(
        "Signed residuals are computed as Y - P(Y=1|X). Divergence between group distributions indicates prediction bias."
    )

    non_white = signed_residual.get("non_white", {})
    white = signed_residual.get("white", {})
    ks_stat = signed_residual.get("ks_statistic")
    ks_pvalue = signed_residual.get("ks_pvalue")

    residual_rows = [
        {
            "Group": "Non-White",
            "mean_residual": non_white.get("mean"),
            "median_cdf_at_0": non_white.get("median_cdf_at_0"),
            "lower_q_alpha_0_1": non_white.get("lower_q_alpha_0_1"),
            "upper_q_alpha_0_1": non_white.get("upper_q_alpha_0_1"),
        },
        {
            "Group": "White",
            "mean_residual": white.get("mean"),
            "median_cdf_at_0": white.get("median_cdf_at_0"),
            "lower_q_alpha_0_1": white.get("lower_q_alpha_0_1"),
            "upper_q_alpha_0_1": white.get("upper_q_alpha_0_1"),
        },
    ]
    st.dataframe(pd.DataFrame(residual_rows), use_container_width=True)

    if ks_stat is not None and ks_pvalue is not None:
        verdict = "bias drift likely" if ks_pvalue < 0.05 else "no significant group-drift detected"
        st.caption(f"KS statistic={ks_stat:.4f}, p-value={ks_pvalue:.6f} -> {verdict}")
