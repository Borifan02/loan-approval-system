"""Streamlit Web Application: Loan Approval Prediction & Explainable AI.

This module provides a professional, accessible, and responsive user interface
for predicting loan approval outcomes and visualizing SHAP feature attributions.
It consumes the decoupled core prediction engine from `src.loan_approval`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src directory to system path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
import streamlit as st

from loan_approval import (
    CONFIDENCE_DISCLAIMER,
    ExplanationResult,
    PredictionResult,
    explain_prediction,
    load_artifacts,
    predict_loan_approval,
    render_waterfall_plot,
    validate_applicant_data,
)
from loan_approval.config import FINAL_MODEL_FEATURES, MODELS_DIR


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval AI — Decision Support System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 20px;
    }
    .approved-badge {
        background-color: #D1FAE5;
        color: #065F46;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .rejected-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Resource Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Machine Learning Models...")
def get_system_artifacts():
    """Caches model and scaler artifacts across Streamlit sessions."""
    return load_artifacts()


def safe_display_dataframe(df: pd.DataFrame, use_container_width: bool = True) -> None:
    """Displays a DataFrame with st.dataframe, falling back to st.table if pyarrow fails."""
    try:
        st.dataframe(df, use_container_width=use_container_width)
    except Exception:
        st.table(df)


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    # Load model artifacts
    try:
        artifacts = get_system_artifacts()
    except Exception as exc:
        st.error(f"❌ Failed to load model artifacts: {exc}")
        st.stop()

    # Sidebar Information & Navigation
    with st.sidebar:
        image_path = ROOT_DIR / "Loan_Prediction_Image.jpg"
        if image_path.is_file():
            st.image(str(image_path), use_container_width=True)

        st.markdown("### 🏦 Loan Decision AI")
        st.caption("Production Decision Support & Explainability Engine")

        st.markdown("---")
        st.markdown("#### ⚙️ System Status")
        st.success("🟢 ML Inference Engine: Online")
        st.info("🟢 SHAP Explainer: Ready")
        st.caption(f"Model: Random Forest ({artifacts.metadata.get('n_estimators', 150)} trees)")

        st.markdown("---")
        st.markdown("#### 📖 Quick Links")
        st.markdown("- [Kaggle Dataset](https://www.kaggle.com/datasets)")
        st.markdown("- [GitHub Project](https://github.com/Borifan02/loan-approval-system)")
        st.markdown("- [Agensi Skill Guide](file:///home/borifan/loan-approval-system-main/agensi-skill/SKILL.md)")

    # Main Tabs
    tab_eval, tab_model, tab_about = st.tabs(
        ["📋 Applicant Evaluation", "📊 Model & Dataset Analytics", "⚖️ Ethics & Responsible AI"]
    )

    # -------------------------------------------------------------------------
    # TAB 1: Evaluation Form & Prediction Results
    # -------------------------------------------------------------------------
    with tab_eval:
        st.markdown('<div class="main-header">Loan Approval Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Evaluate applicant creditworthiness using a trained Random Forest model '
            'with real-time SHAP feature attribution.</div>',
            unsafe_allow_html=True,
        )

        # Form Layout
        with st.form("loan_application_form"):
            col_personal, col_employment = st.columns(2)

            with col_personal:
                st.subheader("1. Applicant Profile")
                gender = st.selectbox("Gender", ["Male", "Female"], help="Demographic field (not used in credit decision)")
                married = st.selectbox("Marital Status", ["Yes", "No"], help="Demographic field (not used in credit decision)")
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], help="Number of financial dependents")
                education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], help="Highest education attained")

            with col_employment:
                st.subheader("2. Employment & Income")
                self_employed = st.selectbox("Self Employed", ["No", "Yes"])
                applicant_income = st.number_input(
                    "Monthly Applicant Income (USD)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=5000.0,
                    step=250.0,
                    help="Primary monthly gross income",
                )
                coapplicant_income = st.number_input(
                    "Monthly Coapplicant Income (USD)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=1500.0,
                    step=250.0,
                    help="Additional household co-borrower monthly income",
                )
                property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

            st.markdown("---")
            col_loan, col_credit = st.columns(2)

            with col_loan:
                st.subheader("3. Loan Request Details")
                loan_amount = st.number_input(
                    "Requested Loan Amount (in thousands USD)",
                    min_value=1.0,
                    max_value=2000.0,
                    value=150.0,
                    step=10.0,
                    help="Loan amount requested in thousands (e.g. 150 = $150,000)",
                )
                loan_term = st.select_slider(
                    "Repayment Term (months)",
                    options=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                    value=360,
                    help="Total loan duration in months (360 months = 30 years)",
                )

            with col_credit:
                st.subheader("4. Credit & History Guidelines")
                credit_history = st.radio(
                    "Credit History Meets Guidelines",
                    options=[1, 0],
                    format_func=lambda x: "Yes (Repayment guidelines met - Score 1)" if x == 1 else "No (Delinquencies or unverified - Score 0)",
                    index=0,
                    help="Repayment history according to standard credit assessment guidelines",
                )

            submit_button = st.form_submit_button("⚡ Evaluate Loan Application", use_container_width=True)

        # Process Submission
        if submit_button:
            applicant_payload = {
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "Property_Area": property_area,
            }

            with st.spinner("Processing applicant features and calculating SHAP values..."):
                pred_result = predict_loan_approval(applicant_payload)
                expl_result = explain_prediction(applicant_payload)

            if not pred_result.success:
                st.error(f"❌ Application Error: {pred_result.error}")
            else:
                st.markdown("---")
                st.markdown("### 🎯 Decision Support Outcome")

                # Results Summary Cards
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)

                with res_col1:
                    if pred_result.prediction == "Approved":
                        st.markdown('<div class="approved-badge">✅ Approved</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="rejected-badge">❌ Rejected</div>', unsafe_allow_html=True)

                with res_col2:
                    st.metric("Model Confidence", f"{pred_result.confidence * 100:.1f}%")

                with res_col3:
                    st.metric("P(Approval)", f"{pred_result.probabilities['approved'] * 100:.1f}%")

                with res_col4:
                    st.metric("P(Rejection)", f"{pred_result.probabilities['rejected'] * 100:.1f}%")

                # Metrics Breakdown
                st.markdown("#### 📐 Financial Metrics Breakdown")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Total Household Income", f"${pred_result.engineered_metrics['Total_Income']:,.2f}")
                m_col2.metric("Estimated Monthly EMI", f"${pred_result.engineered_metrics['Monthly_EMI']:,.2f}")
                m_col3.metric("Net Balance Income Buffer", f"${pred_result.engineered_metrics['Balance_Income']:,.2f}")

                # SHAP Interpretability Section
                st.markdown("---")
                st.markdown("### 🔍 Explainable AI (SHAP Insights)")
                st.info(f"**Decision Summary:** {expl_result.human_readable_summary}")

                plot_col, factor_col = st.columns([3, 2])

                with plot_col:
                    st.markdown("##### Feature Impact on Approval Probability")
                    fig = render_waterfall_plot(expl_result)
                    st.pyplot(fig, clear_figure=True)

                with factor_col:
                    st.markdown("##### Key Positive Drivers (+)")
                    if expl_result.positive_factors:
                        for f in expl_result.positive_factors:
                            st.write(f"🟢 **{f['display_name']}**: {f['summary']} (+{f['impact_magnitude']:.3f})")
                    else:
                        st.write("None observed.")

                    st.markdown("##### Key Risk Drivers (-)")
                    if expl_result.negative_factors:
                        for f in expl_result.negative_factors:
                            st.write(f"🔴 **{f['display_name']}**: {f['summary']} (-{f['impact_magnitude']:.3f})")
                    else:
                        st.write("None observed.")

                # Disclaimer
                st.markdown(
                    f'<div class="disclaimer-box">⚠️ <b>Important Legal Notice:</b> {CONFIDENCE_DISCLAIMER}</div>',
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # TAB 2: Model & Dataset Analytics
    # -------------------------------------------------------------------------
    with tab_model:
        st.markdown("### 📊 Model Architecture & Performance")
        st.write(
            "The production system utilizes a **Random Forest Classifier** trained and cross-validated on the "
            "Kaggle Loan Prediction dataset. It uses 150 ensemble estimators with optimized tree depth and split criteria."
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Model Specification")
            st.json(artifacts.metadata)

        with col_m2:
            st.markdown("#### Global Feature Importances")
            expl_sample = explain_prediction(
                {"ApplicantIncome": 5000, "CoapplicantIncome": 0, "LoanAmount": 100, "Loan_Amount_Term": 360, "Credit_History": 1}
            )
            imp_df = pd.DataFrame(
                list(expl_sample.global_importance.items()),
                columns=["Feature", "Gini Importance"],
            ).sort_values(by="Gini Importance", ascending=False)
            safe_display_dataframe(imp_df, use_container_width=True)

        # Training Dataset Preview if available
        train_csv_path = ROOT_DIR / "data" / "Loan_dataset_train.csv"
        if not train_csv_path.is_file():
            train_csv_path = ROOT_DIR / "Data" / "Loan_dataset_train.csv"
        if train_csv_path.is_file():
            st.markdown("---")
            st.markdown("#### 📂 Training Data Overview")
            df_train = pd.read_csv(train_csv_path)
            st.write(f"Total training samples: **{len(df_train)}** records across **{len(df_train.columns)}** variables.")
            safe_display_dataframe(df_train.head(10), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Ethics & Responsible AI
    # -------------------------------------------------------------------------
    with tab_about:
        st.markdown("### ⚖️ Fair Lending & Ethical AI Architecture")
        st.markdown(
            """
            #### 1. Demographic Feature Exclusion Policy
            In compliance with fair lending principles (such as the **Equal Credit Opportunity Act - ECOA**),
            this model **deliberately excludes demographic attributes** from its prediction formula:
            - **Gender** is excluded to mitigate gender-based credit disparities.
            - **Marital Status** is excluded to avoid discriminating against single or non-traditional households.
            - **Education & Self-Employed status** are excluded to prevent socioeconomic educational bias.
            - **Property Area** is excluded to avoid location-based redlining risks.

            #### 2. Features Utilized
            The active decision model evaluates strictly **financial capacity and repayment history**:
            1. `Credit_History`: Past record of adherence to credit guidelines.
            2. `Total_Income_log`: Log-transformed combined household income.
            3. `EMI_log`: Log-transformed monthly repayment obligation.
            4. `Balance_Income_log`: Net disposable financial buffer remaining after debt service.

            #### 3. Limitations & Disclaimers
            - This application is an educational and analytical tool designed to demonstrate explainable AI techniques.
            - It does **not** replace licensed underwriting processes or human credit analysts.
            - No applicant data entered into this interface is saved, logged to disk, or transmitted to third parties.
            """
        )


if __name__ == "__main__":
    main()
