import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
from preload_explainers import preload_explainers, load_background_data, PredictProbaWrapper, EnsembleProbaWrapper, preprocess_single_transaction
from data_loader import load_transaction_data, preprocess_data, KerasBinaryClassifier, VotingClassifier
from model_explainer import explain_prediction, explain_with_lime, plot_lime_explanation, explain_model
import os
import streamlit.components.v1 as components
import pickle

# Set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💰",
    layout="wide"
)

# Load data with caching
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the data."""
    try:
        df = load_transaction_data()
        df_processed, column_mapping = preprocess_data(df)
        feature_names = [col for col in df_processed.columns if col != 'fraud']
        return df_processed, feature_names, column_mapping
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def verify_model_loading():
    """Verify model loading works independently."""
    st.subheader("Model Verification")
    
    try:
        # Test loading each model
        st.write("Attempting to load Random Forest model...")
        rf_model = joblib.load('models/rf_model.joblib')
        st.success("Random Forest loaded successfully!")
        
        st.write("Attempting to load Neural Network model...")
        nn_model = load_model('models/nn_model.h5')
        st.success("Neural Network loaded successfully!")
        
        st.write("Attempting to load Ensemble model...")
        ensemble_model = joblib.load('models/ensemble_model.joblib')
        st.success("Ensemble model loaded successfully!")
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.error("Possible solutions:")
        st.error("1. Check the model files exist in the 'models' directory")
        st.error("2. Ensure you're using compatible library versions")
        st.error("3. Try retraining the models if files are corrupted")

def create_transaction_input(df_processed, column_mapping):
    """Create input fields for transaction data."""
    st.sidebar.header("Transaction Details")
    
    # Initialize empty dictionary for transaction
    transaction = {}
    
    # Create input fields for each feature
    for col in df_processed.columns:
        if col != 'fraud':
            if col in column_mapping:
                # Handle categorical features
                options = df_processed[col].unique()
                transaction[col] = st.sidebar.selectbox(
                    f"Select {col}",
                    options=options
                )
            else:
                # Handle numerical features
                min_val = float(df_processed[col].min())
                max_val = float(df_processed[col].max())
                transaction[col] = st.sidebar.slider(
                    f"Select {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=float(df_processed[col].median())
                )
    
    return pd.DataFrame([transaction])

def plot_shap_summary(model, X, feature_names):
    """Create SHAP summary plot."""
    try:
        # Create explainer based on model type
        if hasattr(model, 'predict_proba'):
            # For scikit-learn models
            explainer = shap.TreeExplainer(model)
        else:
            # For Keras models
            background_data = shap.utils.sample(X, 100)
            explainer = shap.KernelExplainer(model.predict, background_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handle different SHAP values formats
        if isinstance(shap_values, list):
            # For multi-class models, use the positive class (index 1)
            if len(shap_values) > 1:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False
        )
        plt.title('SHAP Summary Plot for Fraud Predictions')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"An error occurred while generating SHAP summary: {str(e)}")
        return None

def plot_shap_dependence(model, X, feature_names, feature_idx):
    """Create SHAP dependence plot for a specific feature."""
    try:
        # Create explainer based on model type
        if hasattr(model, 'predict_proba'):
            # For scikit-learn models
            explainer = shap.TreeExplainer(model)
        else:
            # For Keras models
            background_data = shap.utils.sample(X, 100)
            explainer = shap.KernelExplainer(model.predict, background_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handle different SHAP values formats
        if isinstance(shap_values, list):
            # For multi-class models, use the positive class (index 1)
            if len(shap_values) > 1:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        
        # Create dependence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=feature_names,
            show=False
        )
        plt.title(f'SHAP Dependence Plot for {feature_names[feature_idx]}')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"An error occurred while generating SHAP dependence plot: {str(e)}")
        return None

def load_models():
    """Load the trained models and preloaded explainers from disk"""
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            st.error("Models directory not found. Please train the models first.")
            return None, None, None, None

        # First load the Random Forest to get feature count
        rf_path = os.path.join(models_dir, 'rf_model.joblib')
        if not os.path.exists(rf_path):
            st.error("Random Forest model not found. Please train the models first.")
            return None, None, None, None

        try:
            rf_model = joblib.load(rf_path)
            n_features = rf_model.n_features_in_
            st.success("Random Forest model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading Random Forest model: {str(e)}")
            return None, None, None, None

        # Now load other models with proper feature count
        models = {
            'random_forest': rf_model,
            'neural_network': None,
            'ensemble': None
        }

        # Load Neural Network
        nn_path = os.path.join(models_dir, 'nn_model.h5')
        if os.path.exists(nn_path):
            try:
                keras_model = load_model(nn_path)
                # Initialize with correct feature count
                nn_model = KerasBinaryClassifier(
                    model=keras_model,
                    n_features=n_features
                )
                # Perform dummy fit to set attributes
                nn_model.fit(np.zeros((1, n_features)))
                models['neural_network'] = nn_model
                st.success("Neural Network model loaded and initialized successfully!")
            except Exception as e:
                st.error(f"Error loading Neural Network model: {str(e)}")
                return None, None, None, None
        else:
            st.error("Neural Network model not found. Please train the models first.")
            return None, None, None, None

        # Load Ensemble
        ensemble_path = os.path.join(models_dir, 'ensemble_model.joblib')
        if os.path.exists(ensemble_path):
            try:
                ensemble_model = joblib.load(ensemble_path)
                # Ensure feature count is set
                if isinstance(ensemble_model, VotingClassifier):
                    ensemble_model._n_features_in = n_features
                models['ensemble'] = ensemble_model
                st.success("Ensemble model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading Ensemble model: {str(e)}")
                return None, None, None, None
        else:
            st.error("Ensemble model not found. Please train the models first.")
            return None, None, None, None

        # Try to load explainers cache
        explainers = {}
        explainers_path = os.path.join(models_dir, 'explainers_cache.pkl')
        if os.path.exists(explainers_path):
            try:
                with open(explainers_path, 'rb') as f:
                    explainers = pickle.load(f)
                st.success("Preloaded explainers loaded successfully!")
            except Exception as e:
                st.warning(f"Could not load explainers cache: {str(e)}")

        return (
            models['ensemble'],
            models['neural_network'],
            models['random_forest'],
            explainers
        )

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def main():
    st.title("💰 Fraud Detection System")
    
    # Sidebar for verification or transaction input
    if st.sidebar.button("Verify Model Loading"):
        verify_model_loading()
        return

    # Model & visualization selectors
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=["Random Forest", "Neural Network", "Ensemble"]
    )
    viz = st.sidebar.selectbox(
        "Select Explanation Method",
        options=["SHAP", "LIME"]
    )

    # Load data
    df_processed, feature_names, column_mapping = load_and_preprocess_data()
    if df_processed is None:
        return

    # Load models & explainers
    ensemble_model, nn_model, rf_model, explainers = load_models()
    if None in [ensemble_model, nn_model, rf_model]:
        return

    models = {
        "Random Forest": rf_model,
        "Neural Network": nn_model,
        "Ensemble": ensemble_model
    }
    model = models[model_name]
    expl_key = model_name.lower().replace(" ", "_")
    
    # Build transaction input
    transaction_df = create_transaction_input(df_processed, column_mapping)
    st.subheader("Transaction Details")
    st.dataframe(transaction_df)

    if not st.button("Predict & Explain"):
        return

    # Preprocess & predict
    X = preprocess_single_transaction(transaction_df.iloc[0].to_dict(), column_mapping)
    X = X.reshape(1, -1)
    st.write("Features:", X)

    if model_name == "Neural Network":
        prob = float(model.predict(X)[0])
        pred = int(prob > 0.5)
        probs = [1 - prob, prob]
    else:
        probs = model.predict_proba(X)[0]
        pred = int(model.predict(X)[0])

    if pred == 1:
        st.error(f"⚠️ Fraud (P={probs[1]:.3f})")
    else:
        st.success(f"✅ Legitimate (P={probs[0]:.3f})")

    st.subheader(f"Explanation ({viz})")

    if viz == "SHAP":
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Force Plot", "Waterfall Plot", "Bar Plot", "Dependence Plot"]
        )

        with tab1:
            fig = explain_prediction(
                expl_key, X, feature_names, explainers, plot_type="force"
            )
            st.pyplot(fig)

        with tab2:
            fig = explain_prediction(
                expl_key, X, feature_names, explainers, plot_type="waterfall"
            )
            st.pyplot(fig)

        with tab3:
            fig = explain_prediction(
                expl_key, X, feature_names, explainers, plot_type="bar"
            )
            st.pyplot(fig)

        # with tab4:
        #     feature = st.selectbox("Feature for dependence", feature_names)
        #     fig = explain_model(
        #         models[model_name],
        #         df_processed.drop("fraud", axis=1).values,
        #         feature_names,
        #         plot_type="dependence",
        #         dependence_feature=feature
        #     )
        #     st.pyplot(fig)

    # else:  # LIME
    #     lime_exp = explain_with_lime(
    #         model, X, feature_names
    #     )
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     lime_exp.as_pyplot_figure()
    #     # st.pyplot(fig)
    #     st.write(lime_exp.as_list())


if __name__ == "__main__":
    main() 