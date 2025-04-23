import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
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
    
    # Add model verification button
    if st.sidebar.button("Verify Model Loading"):
        verify_model_loading()
        return
    
    # Sidebar for model and visualization selection
    st.sidebar.header("Model and Visualization Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        options=["Random Forest", "Neural Network", "Ensemble"]
    )
    
    # Visualization selection
    visualization_type = st.sidebar.selectbox(
        "Select Visualization Tool",
        options=["SHAP", "LIME"]
    )
    
    # Load data
    df_processed, feature_names, column_mapping = load_and_preprocess_data()
    if df_processed is None:
        st.error("Failed to load data. Please check the data file.")
        return
    
    # Load models and explainers
    ensemble_model, nn_model, rf_model, explainers = load_models()
    if None in [ensemble_model, nn_model, rf_model]:
        st.error("Failed to load models. Please check the model files.")
        return
    
    # Create a dictionary of models
    models = {
        "Random Forest": rf_model,
        "Neural Network": nn_model,
        "Ensemble": ensemble_model
    }
    
    # Select the appropriate model
    model = models[model_type]
    
    # Create transaction input
    transaction_df = create_transaction_input(df_processed, column_mapping)
    
    # Display transaction details
    st.subheader("Transaction Details")
    st.dataframe(transaction_df)
    
    # Make prediction
    if st.button("Predict"):
        # Preprocess transaction
        transaction_values = transaction_df.values
        
        # Debug: Print transaction values
        st.write("Transaction values:", transaction_values)
        
        # Get prediction and probability based on model type
        if model_type == "Neural Network":
            # Neural Network returns a single probability
            probability = float(model.predict(transaction_values, verbose=0)[0])
            prediction = 1 if probability > 0.5 else 0
            fraud_probability = probability
            legitimate_probability = 1 - probability
        else:
            # Random Forest and Ensemble return probability array
            prediction = model.predict(transaction_values)[0]
            probabilities = model.predict_proba(transaction_values)[0]
            fraud_probability = float(probabilities[1])
            legitimate_probability = float(probabilities[0])
        
        # Display prediction
        st.subheader("Prediction")
        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction (Probability: {fraud_probability:.4f})")
        else:
            st.success(f"✅ Legitimate Transaction (Probability: {legitimate_probability:.4f})")
        
        # Display model information
        st.subheader(f"Model: {model_type}")
        st.write(f"Using {visualization_type} for explanation")
        
        if visualization_type == "SHAP":
            # Create tabs for different SHAP explanations
            tab1, tab2, tab3, tab4 = st.tabs([
                "Force Plot", 
                "Waterfall Plot", 
                "Summary Plot",
                "Dependence Plot"
            ])
            
            st.subheader("SHAP Force Plot")
            if explainers and model_type.lower() in explainers:
                # Use preloaded explainer if available
                shap_values = explainers[model_type.lower()].shap_values(transaction_values)
                fig = plt.figure()
                shap.force_plot(
                    explainers[model_type.lower()].expected_value,
                    shap_values[0] if isinstance(shap_values, list) else shap_values,
                    transaction_values[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig)
            else:
                # Fallback to original method
                force_plot = explain_prediction(
                    model,
                    transaction_values,
                    feature_names,
                    plot_type='force'
                )
                st.pyplot(force_plot)
            
            with tab2:
                st.subheader("SHAP Waterfall Plot")
                waterfall_plot = explain_prediction(
                    model,
                    transaction_values,
                    feature_names,
                    plot_type='waterfall'
                )
                st.pyplot(waterfall_plot)
            
            with tab3:
                st.subheader("SHAP Summary Plot")
                summary_plot = plot_shap_summary(
                    model,
                    df_processed.drop('fraud', axis=1).values,
                    feature_names
                )
                st.pyplot(summary_plot)
            
            with tab4:
                st.subheader("SHAP Dependence Plot")
                selected_feature = st.selectbox(
                    "Select feature for dependence plot",
                    options=feature_names
                )
                dependence_plot = plot_shap_dependence(
                    model,
                    df_processed.drop('fraud', axis=1).values,
                    feature_names,
                    feature_names.index(selected_feature)
                )
                st.pyplot(dependence_plot)
        
        else:  # LIME visualization
            st.subheader("LIME Explanation")
            lime_explanation = explain_with_lime(
                model,
                transaction_values,
                feature_names
            )
            
            # Create a figure for LIME plot
            fig, ax = plt.subplots(figsize=(10, 6))
            lime_explanation.as_pyplot_figure()
            plt.title('LIME Explanation for Transaction Prediction')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display LIME explanation details
            st.write("Feature Contributions:")
            for feature, weight in lime_explanation.as_list():
                st.write(f"{feature}: {weight:.4f}")

if __name__ == "__main__":
    main() 