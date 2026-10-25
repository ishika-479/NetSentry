import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# NetSentry: Configuration
st.set_page_config(
    page_title="NetSentry | Network Traffic Analysis",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "netsentry_model.joblib"
FEATURES_PATH = BASE_DIR / "netsentry_features.json"

st.title("NetSentry")
st.subheader("AI-powered Network Traffic Classification")
st.write(
    "Upload a network-flow CSV to classify records as BENIGN "
    "or DDoS using a trained Random Forest model."
)

st.warning(
    "Research prototype: predictions are not proof of an attack "
    "and should not be used as the sole basis for security decisions."
)

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)
    return model, features


if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error(
        "Model files are missing. Add netsentry_model.joblib "
        "and netsentry_features.json to the repository."
    )
    st.stop()

try:
    model, feature_names = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

if not isinstance(feature_names, list) or not feature_names:
    st.error("The feature list is empty or invalid.")
    st.stop()

st.caption(f"Loaded model with {len(feature_names)} expected features.")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload a network-traffic CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

    # Normalize column names
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        st.error("The uploaded CSV has no records.")
        st.stop()

    # Remove label column from prediction inputs, if present
    input_df = df.drop(
        columns=["Label", "label"],
        errors="ignore",
    ).copy()

    # Check required features
    missing_features = [
        feature for feature in feature_names
        if feature not in input_df.columns
    ]

    if missing_features:
        st.error(
            f"The CSV is missing {len(missing_features)} "
            "features required by this model."
        )
        st.write("Missing feature names:")
        st.code("\n".join(missing_features))
        st.info(
            "Use a CSV from the same dataset and feature schema "
            "as the file used to train NetSentry."
        )
        st.stop()

    # Preserve the exact training feature order
    X = input_df.loc[:, feature_names].copy()

    # Ensure all model inputs are numeric
    for column in feature_names:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # The saved pipeline handles missing numeric values
    try:
        predictions = model.predict(X)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    # Map model outputs to human-readable labels
    label_map = {0: "BENIGN", 1: "DDoS"}

    if not set(np.unique(predictions)).issubset(label_map):
        st.error("The model returned unexpected class labels.")
        st.stop()

    result_df = df.copy()
    result_df["NetSentry Prediction"] = [
        label_map[int(value)] for value in predictions
    ]

    # -------------------------------
    # Results dashboard
    # -------------------------------
    benign_count = int((predictions == 0).sum())
    ddos_count = int((predictions == 1).sum())
    total_count = len(predictions)

    st.divider()
    st.header("Analysis Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Records analyzed", f"{total_count:,}")
    col2.metric("Predicted BENIGN", f"{benign_count:,}")
    col3.metric("Predicted DDoS", f"{ddos_count:,}")

    st.caption(
        "These are model predictions, not independently verified "
        "attack counts."
    )

    chart_data = pd.DataFrame({
        "Prediction": ["BENIGN", "DDoS"],
        "Records": [benign_count, ddos_count],
    }).set_index("Prediction")

    st.subheader("Predicted Traffic Distribution")
    st.bar_chart(chart_data)

    st.subheader("Prediction Results")
    st.dataframe(result_df.head(100), use_container_width=True)

    st.download_button(
        "Download all predictions as CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="netsentry_predictions.csv",
        mime="text/csv",
    )

    # Optional evaluation if true labels are included
    label_column = next(
        (column for column in df.columns
         if column.lower() == "label"),
        None,
    )

    if label_column:
        actual = df[label_column].astype(str).str.strip().str.upper()
        known_labels = actual.isin(["BENIGN", "DDOS"])

        if known_labels.any():
            y_true = actual[known_labels].map(
                {"BENIGN": 0, "DDOS": 1}
            )
            y_hat = predictions[known_labels.to_numpy()]

            from sklearn.metrics import (
                accuracy_score,
                classification_report,
            )

            st.subheader("Evaluation Against Provided Labels")
            st.write(
                "Only records labeled BENIGN or DDoS are included."
            )
            st.metric(
                "Accuracy",
                f"{accuracy_score(y_true, y_hat):.2%}",
            )
            st.text(
                classification_report(
                    y_true,
                    y_hat,
                    labels=[0, 1],
                    target_names=["BENIGN", "DDoS"],
                    zero_division=0,
                )
            )

st.divider()
st.caption(
    "NetSentry is an educational machine-learning prototype. "
    "It does not monitor live network traffic."
)
