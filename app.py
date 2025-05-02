import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from lime import lime_tabular
from data_loader import load_transaction_data, preprocess_data, KerasBinaryClassifier
from utils import load_column_mapping, preprocess_single_transaction
from model_explainer import explain_prediction, explain_with_lime, plot_lime_explanation, get_explainer
from preload_explainers import load_background_data, PredictProbaWrapper, EnsembleProbaWrapper

import os

# Streamlit page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💰",
    layout="wide"
)

@st.cache_data
def load_and_preprocess_data():
    df = load_transaction_data()
    df_processed, column_mapping = preprocess_data(df)
    feature_names = [c for c in df_processed.columns if c != 'fraud']
    return df_processed, feature_names, column_mapping

@st.cache_resource
def load_models(models_dir='models'):
    # Load Random Forest
    rf_path = os.path.join(models_dir, 'rf_model.joblib')
    rf_model = joblib.load(rf_path)
    n_features = rf_model.n_features_in_
    # Load Neural Network
    nn_path = os.path.join(models_dir, 'nn_model.h5')
    keras_net = load_model(nn_path)
    nn_model = KerasBinaryClassifier(model=keras_net, n_features=n_features)
    nn_model.fit(np.zeros((1, n_features)))
    # Load ensemble variants
    ensemble_models = {}
    for kind in ['voting', 'adaboost', 'stacking']:
        path = os.path.join(models_dir, f'ensemble_{kind}.joblib')
        ens = joblib.load(path)
        # ensure feature count
        if hasattr(ens, '_n_features_in'):
            ens._n_features_in = n_features
        ensemble_models[kind] = ens
    models = {
        'Random Forest': rf_model,
        'Neural Network': nn_model,
        'Voting Ensemble': ensemble_models['voting'],
        'AdaBoost Ensemble': ensemble_models['adaboost'],
        'Stacked Ensemble': ensemble_models['stacking'],
    }
    return models

@st.cache_data
def get_background(n_samples=50, n_features=None):
    return load_background_data(n_samples, n_features)

@st.cache_resource
def get_shap_explainer_for(model_name, _model):
    # Note: leading underscore on _model tells Streamlit not to hash the model object
    # prepare background
    n_features = getattr(_model, 'n_features_in_', None)
    background = get_background(n_samples=50, n_features=n_features)
    # wrap for voting/stacking
    if model_name in ['Voting Ensemble', 'AdaBoost Ensemble', 'Stacked Ensemble']:
        # Check if tree-based or require wrapper
        if hasattr(_model, 'estimators_') or hasattr(_model, 'base_estimator_'):
            expl_model = _model
        else:
            expl_model = EnsembleProbaWrapper(_model)
    elif model_name == 'Neural Network':
        expl_model = PredictProbaWrapper(_model)
    else:
        expl_model = _model
    return get_explainer(expl_model, background)


def create_transaction_input(df_processed, column_mapping):
    st.sidebar.header("Transaction Details")
    inputs = {}
    for col in df_processed.columns:
        if col == 'fraud': continue
        if col in column_mapping:
            opts = list(column_mapping[col].keys())
            inputs[col] = st.sidebar.selectbox(col, opts)
        else:
            mi, ma = float(df_processed[col].min()), float(df_processed[col].max())
            inputs[col] = st.sidebar.slider(col, mi, ma, float(df_processed[col].median()))
    return pd.DataFrame([inputs])


def main():
    st.title("💰 Fraud Detection System")

    # Load data
    df_processed, feature_names, column_mapping = load_and_preprocess_data()
    if df_processed is None:
        return

    # Load models (cached)
    models = load_models()

    # Sidebar controls
    model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
    method = st.sidebar.selectbox("Explanation Method", ["SHAP", "LIME"])  

    # Transaction input
    transaction_df = create_transaction_input(df_processed, column_mapping)
    st.subheader("Transaction Details")
    st.dataframe(transaction_df)

    if not st.button("Predict & Explain"):
        return

    # Preprocess for prediction
    proc = preprocess_single_transaction(transaction_df.iloc[0].to_dict(), column_mapping)
    X = np.array(proc, dtype=np.float32).reshape(1, -1)

    # Select model
    model = models[model_choice]
    # Predict
    if model_choice == 'Neural Network':
        prob = float(model.predict(X)[0])
        probs = [1 - prob, prob]
        pred = int(prob > 0.5)
    else:
        probs = model.predict_proba(X)[0]
        pred = int(model.predict(X)[0])

    # Show outcome
    if pred == 1:
        st.error(f"⚠️ Fraud (P={probs[1]:.3f})")
    else:
        st.success(f"✅ Legitimate (P={probs[0]:.3f})")

    st.subheader(f"Explanation ({method}) for {model_choice}")

    if method == "SHAP":
        # Get SHAP explainer (cached)
        expl = get_shap_explainer_for(model_choice, model)
        expl_key = model_choice.lower().replace(' ', '_')
        # Show three plot types
        tab1, tab2, tab3 = st.tabs(["Force Plot", "Waterfall", "Bar"])
        with tab1:
            fig = explain_prediction(
                model_name=expl_key,
                transaction=X,
                feature_names=feature_names,
                explainers={expl_key: expl},
                plot_type="force"
            )
            st.pyplot(fig)
        with tab2:
            fig = explain_prediction(
                model_name=expl_key,
                transaction=X,
                feature_names=feature_names,
                explainers={expl_key: expl},
                plot_type="waterfall"
            )
            st.pyplot(fig)
        with tab3:
            fig = explain_prediction(
                model_name=expl_key,
                transaction=X,
                feature_names=feature_names,
                explainers={expl_key: expl},
                plot_type="bar"
            )
            st.pyplot(fig)
    else:
        # LIME explanation
        lime_exp = explain_with_lime(model, X, feature_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        lime_exp.as_pyplot_figure()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
