
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
from scipy import sparse

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background-color: #111b2e;
    }
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #60a5fa;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 24px;
    }
    div[data-testid="stMetric"] {
        background: #172338;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #263650;
    }
    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        min-height: 42px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD SAVED ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
    BASE_DIR / "full_dataset_fraud_model.keras",
        compile=False
    )

    preprocessor = joblib.load(
        BASE_DIR / "full_dataset_preprocessor.pkl"
    )

    with open(BASE_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)

    with open(BASE_DIR / "full_dataset_threshold.json") as f:
        threshold_data = json.load(f)


    if isinstance(threshold_data, dict):
        threshold = float(threshold_data["threshold"])
    else:
        threshold = float(threshold_data)

    return model, preprocessor, feature_columns, threshold


try:
    model, preprocessor, feature_columns, threshold = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the saved model artifacts: {exc}")
    st.info("Check that all four files exist inside fraud_detection_model/.")
    st.stop()


# --------------------------------------------------
# IDENTIFY CATEGORICAL FIELDS AND THEIR OPTIONS
# --------------------------------------------------

def find_categorical_options(preprocessor):
    options = {}

    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or transformer == "passthrough":
            continue

        if isinstance(columns, str):
            columns = [columns]
        elif isinstance(columns, slice):
            continue
        else:
            columns = list(columns)

        encoder = None

        if hasattr(transformer, "categories_"):
            encoder = transformer
        elif hasattr(transformer, "named_steps"):
            for step in transformer.named_steps.values():
                if hasattr(step, "categories_"):
                    encoder = step

        if encoder is not None:
            for col, categories in zip(columns, encoder.categories_):
                options[col] = list(categories)

    return options


categorical_options = find_categorical_options(preprocessor)
categorical_columns = set(categorical_options.keys())


# --------------------------------------------------
# SESSION STATISTICS
# --------------------------------------------------

if "total" not in st.session_state:
    st.session_state.total = 0
    st.session_state.fraud = 0
    st.session_state.legitimate = 0
    st.session_state.history = []


def record_predictions(probabilities, predictions):
    st.session_state.total += len(predictions)
    st.session_state.fraud += int(np.sum(predictions))
    st.session_state.legitimate += int(len(predictions) - np.sum(predictions))

    new_records = [
        {
            "Fraud Probability (%)": round(float(prob) * 100, 2),
            "Prediction": (
                "Potential Fraud" if int(pred) == 1
                else "Legitimate"
            ),
        }
        for prob, pred in zip(probabilities, predictions)
    ]

    st.session_state.history.extend(new_records)
    st.session_state.history = st.session_state.history[-1000:]


# --------------------------------------------------
# PREPROCESS AND PREDICT
# --------------------------------------------------

def predict_dataframe(data):
    # Enforce original training feature order.
    data = data[feature_columns].copy()

    transformed = preprocessor.transform(data)

    # Process sparse matrices in manageable batches.
    probabilities = []

    for start in range(0, len(data), 2000):
        end = min(start + 2000, len(data))
        batch = transformed[start:end]

        if sparse.issparse(batch):
            batch = batch.toarray()

        batch = np.asarray(batch, dtype=np.float32)

        output = model.predict(batch, verbose=0)
        probabilities.extend(np.asarray(output).reshape(-1).tolist())

    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    return probabilities, predictions


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🛡️ FraudGuard AI")
    st.caption("Intelligent Transaction Monitoring")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Detect Fraud",
            "Bulk Analysis",
            "Model Information",
        ],
    )

    st.divider()
    st.success("ANN model loaded")
    st.caption(f"Decision threshold: {threshold:.4f}")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Overview Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Monitor transaction predictions '
        'from your ANN fraud detection model.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions Analyzed",
        f"{st.session_state.total:,}"
    )
    col2.metric(
        "Potential Fraud",
        f"{st.session_state.fraud:,}"
    )
    col3.metric(
        "Legitimate",
        f"{st.session_state.legitimate:,}"
    )

    fraud_rate = (
        100 * st.session_state.fraud / st.session_state.total
        if st.session_state.total else 0
    )

    col4.metric("Predicted Fraud Rate", f"{fraud_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Prediction Distribution")

        if st.session_state.total:
            chart_data = pd.DataFrame({
                "Category": ["Potential Fraud", "Legitimate"],
                "Count": [
                    st.session_state.fraud,
                    st.session_state.legitimate,
                ],
            })

            fig = px.pie(
                chart_data,
                names="Category",
                values="Count",
                hole=0.55,
                color="Category",
                color_discrete_map={
                    "Potential Fraud": "#ef4444",
                    "Legitimate": "#22c55e",
                },
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a prediction to populate this chart.")

    with right:
        st.subheader("Recent Predictions")

        if st.session_state.history:
            recent = pd.DataFrame(
                st.session_state.history[-10:]
            ).iloc[::-1]

            st.dataframe(
                recent,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No transactions have been analyzed yet.")

    st.caption(
        "Dashboard statistics reflect predictions made during "
        "the current app session, not your complete training dataset."
    )


# --------------------------------------------------
# SINGLE TRANSACTION PREDICTION
# --------------------------------------------------

elif page == "Detect Fraud":

    st.markdown(
        '<div class="main-title">Transaction Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Enter transaction features to '
        'evaluate the likelihood of fraud.</div>',
        unsafe_allow_html=True,
    )

    with st.form("transaction_form"):

        st.subheader("Transaction Details")

        st.caption(
            "Enter the same features used during model training. "
            "Use realistic values and select known categories."
        )

        form_cols = st.columns(2)
        values = {}

        for index, col in enumerate(feature_columns):
            with form_cols[index % 2]:
                if col in categorical_columns:
                    categories = categorical_options[col]

                    if len(categories) > 0:
                        clean_categories = [
                            value.item() if isinstance(value, np.generic)
                            else value
                            for value in categories
                        ]

                        values[col] = st.selectbox(
                            col.replace("_", " ").title(),
                            options=clean_categories,
                            key=f"single_{col}",
                        )
                    else:
                        values[col] = st.text_input(
                            col.replace("_", " ").title(),
                            key=f"single_{col}",
                        )
                else:
                    values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        value=0.0,
                        format="%.6f",
                        key=f"single_{col}",
                    )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            use_container_width=True,
        )

    if submitted:
        try:
            input_df = pd.DataFrame(
                [values],
                columns=feature_columns,
            )

            probabilities, predictions = predict_dataframe(input_df)
            probability = float(probabilities[0])
            prediction = int(predictions[0])

            record_predictions(probabilities, predictions)

            st.divider()
            st.subheader("Prediction Result")

            result_col, probability_col = st.columns(2)

            if prediction == 1:
                result_col.error("Potential Fraud Detected")
                result_col.write(
                    "The model classified this transaction as potential fraud."
                )
            else:
                result_col.success("Classified as Legitimate")
                result_col.write(
                    "The model classified this transaction as legitimate."
                )

            probability_col.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(np.clip(probability, 0, 1)))

            st.caption(
                f"Classification threshold: {threshold:.4f}. "
                "This is a model prediction, not proof of fraud."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# BULK CSV ANALYSIS
# --------------------------------------------------

elif page == "Bulk Analysis":

    st.markdown(
        '<div class="main-title">Bulk Transaction Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Upload a CSV to analyze multiple '
        'transactions at once.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    st.caption(
        "The CSV must contain every feature listed in "
        "feature_columns.json. An actual is_fraud label is not required."
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)

            st.write("Preview of uploaded data")
            st.dataframe(data.head(10), use_container_width=True)

            missing = [
                col for col in feature_columns
                if col not in data.columns
            ]

            if missing:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing)
                )
            elif data.empty:
                st.warning("The uploaded CSV contains no rows.")
            elif st.button(
                "Analyze All Transactions",
                use_container_width=True,
            ):
                with st.spinner("Analyzing transactions..."):

                    results = []

                    for start in range(0, len(data), 2000):
                        batch_df = data.iloc[
                            start:start + 2000
                        ].copy()

                        probs, preds = predict_dataframe(batch_df)

                        batch_df["Fraud Probability (%)"] = (
                            probs * 100
                        ).round(2)

                        batch_df["Prediction"] = np.where(
                            preds == 1,
                            "Potential Fraud",
                            "Legitimate",
                        )

                        results.append(batch_df)

                        record_predictions(probs, preds)

                    output_df = pd.concat(
                        results,
                        ignore_index=True,
                    )

                st.success(
                    f"Analyzed {len(output_df):,} transactions."
                )

                fraud_count = int(
                    (output_df["Prediction"] == "Potential Fraud").sum()
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Analyzed", f"{len(output_df):,}")
                col2.metric("Potential Fraud", f"{fraud_count:,}")
                col3.metric(
                    "Legitimate",
                    f"{len(output_df) - fraud_count:,}",
                )

                st.dataframe(
                    output_df.head(100),
                    use_container_width=True,
                )

                st.download_button(
                    "Download Predictions CSV",
                    data=output_df.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.markdown(
        '<div class="main-title">Model Information</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Details about the saved ANN '
        'and its prediction configuration.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Type", "ANN")
    col2.metric("Input Features", len(feature_columns))
    col3.metric("Decision Threshold", f"{threshold:.4f}")

    st.subheader("Input Features")

    feature_df = pd.DataFrame({
        "Feature": feature_columns,
        "Input Type": [
            "Categorical" if col in categorical_columns
            else "Numeric / Other"
            for col in feature_columns
        ],
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Model Evaluation")

    st.info(
        "Add your final held-out test-set metrics here after confirming "
        "which saved model and threshold produced them. Do not use "
        "training metrics as test metrics."
    )

    st.markdown("""
    **Recommended metrics to report**
    - Precision and recall for the fraud class
    - F1-score
    - PR-AUC and ROC-AUC
    - Confusion matrix
    """)

    st.caption(
        "The classification threshold controls the trade-off between "
        "false alarms and missed fraudulent transactions."
    )
