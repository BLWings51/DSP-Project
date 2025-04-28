import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import streamlit.components.v1 as components
from lime import lime_tabular
import lime
import streamlit as st
import os
import pickle
from preload_explainers import preload_explainers, load_background_data
from utils import load_column_mapping, preprocess_single_transaction, load_model_from_disk

# Cache for explainers
_explainer_cache = {}

def get_explainer(model, background_data=None):
    """
    Return a cached SHAP explainer:
      - RandomForestClassifier → TreeExplainer
      - all other models       → KernelExplainer
    """
    mid = id(model)
    if mid in _explainer_cache:
        return _explainer_cache[mid]

    if background_data is None:
        # 100 samples, trimmed to your model’s feature count
        background_data = load_background_data(100, getattr(model, 'n_features_in_', None))

    if isinstance(model, RandomForestClassifier):
        expl = shap.TreeExplainer(model)
    else:
        predict_fn = getattr(model, 'predict_proba', model)
        expl = shap.KernelExplainer(
            predict_fn,
            background_data,
            output_names=['Not Fraud','Fraud']
        )

    _explainer_cache[mid] = expl
    return expl

def explain_prediction(model_name, transaction, feature_names, explainers, plot_type='force'):
    """
    Generate SHAP explanation for a single transaction prediction using pre-built explainers.
    
    Parameters:
    -----------
    model_name : str
        Name of the model ('random_forest', 'neural_network', or 'ensemble')
    transaction : numpy.ndarray or pandas.Series
        Single transaction to explain
    feature_names : list
        Names of the features
    explainers : dict
        Dictionary of pre-built explainers loaded from cache
    plot_type : str, optional
        Type of SHAP plot to generate ('force', 'waterfall', or 'bar'). Defaults to 'force'
        
    Returns:
    --------
    matplotlib.figure.Figure
        The SHAP explanation plot
    """
    try:
        # Ensure transaction is in the correct format
        if isinstance(transaction, pd.Series):
            transaction = transaction.values
        elif isinstance(transaction, np.ndarray):
            if len(transaction.shape) == 1:
                transaction = transaction.reshape(1, -1)
        else:
            raise ValueError("Transaction must be a pandas Series or numpy array")
            
        # Get the pre-built explainer
        explainer = explainers[model_name]
        
        # Get SHAP values - ensure we pass a 2D array
        print("Transaction shape:", transaction.shape)
        shap_values = explainer(transaction)
        
        # Extract values and expected value based on explainer type
        if hasattr(shap_values, "values"):
            # For ExactExplainer
            vals = shap_values.values
            expected_value = shap_values.base_values
        else:
            # For TreeExplainer or KernelExplainer
            vals = shap_values
            expected_value = explainer.expected_value
        
        # Debug printing
        print(f"Model: {model_name}")
        print(f"SHAP values shape: {vals.shape if hasattr(vals, 'shape') else 'list'}")
        print(f"SHAP values: {vals}")
        
        # Get the positive class values
        if model_name == 'random_forest':
            # TreeExplainer for RF
            if isinstance(vals, list):
                # If vals is a list, it contains values for each class
                single_output = np.array(vals[1]).reshape(1, -1)  # Reshape to 2D array
            elif len(vals.shape) > 1:
                if vals.shape[-1] > 1:  # Multiple classes
                    single_output = vals[..., 1].reshape(1, -1)  # Reshape to 2D array
                else:
                    single_output = vals.reshape(1, -1)  # Reshape to 2D array
            else:
                single_output = vals.reshape(1, -1)  # Reshape to 2D array
        else:
            # Kernel or Permutation or high-level
            if isinstance(vals, list):
                single_output = np.array(vals[1]).reshape(1, -1)  # Reshape to 2D array
            elif len(vals.shape) > 1:
                single_output = vals[..., 1].reshape(1, -1)  # Reshape to 2D array
            else:
                single_output = vals.reshape(1, -1)  # Reshape to 2D array
        
        # Debug printing
        print(f"Single output shape: {single_output.shape}")
        print(f"Single output: {single_output}")
        
        # Get the expected value for the positive class
        if hasattr(expected_value, "__len__"):
            if len(expected_value) > 1:
                expected_value = np.array([expected_value[1]])  # Convert to 1D array
            else:
                expected_value = np.array([expected_value[0]])  # Convert to 1D array
        else:
            expected_value = np.array([expected_value])  # Convert scalar to 1D array
        
        # Debug printing
        print(f"Expected value shape: {expected_value.shape}")
        print(f"Expected value: {expected_value}")

        # Flatten arrays to 1D
        sv = single_output.flatten()  # SHAP values for each feature
        bv = expected_value.flatten()[0]  # Base value (scalar)
        feat_vals = transaction.reshape(-1)  # Feature values
        
        if plot_type == 'force':
            # Create force plot using new SHAP v0.20 API
            force_plot = shap.plots.force(
                bv,  # Base value
                sv,  # SHAP values
                feature_names=feature_names,
                features=feat_vals,
                show=False,
                matplotlib=True
            )
            return force_plot
            
        elif plot_type == 'waterfall':
            # build correct Explanation
            explanation = shap.Explanation(
                values=sv,
                base_values=bv,
                data=feat_vals,
                feature_names=feature_names
            )
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(
                explanation,
                max_display=len(sv),    # show all features without tick errors
                show=False
            )
            fig = plt.gcf()
            plt.tight_layout()
            return fig

            
        elif plot_type == 'bar':
            # Create bar plot
            explanation = shap.Explanation(
                values=sv,
                base_values=bv,
                data=feat_vals,
                feature_names=feature_names
            )
            plt.figure(figsize=(8, 6))
            shap.plots.bar(
                explanation,
                max_display=len(sv),    # show all features without tick errors
                show=False
            )
            fig = plt.gcf()
            plt.tight_layout()
            return fig
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
    except Exception as e:
        raise Exception(f"An error occurred while generating SHAP explanation: {str(e)}")

def explain_model(model, X, feature_names, background_data=None):
    """
    Generate global SHAP explanation for the model.
    
    Parameters:
    -----------
    model : sklearn.estimator
        Trained model to explain
    X : numpy.ndarray or pandas.DataFrame
        Data to explain
    feature_names : list
        List of feature names
    background_data : numpy.ndarray or pandas.DataFrame, optional
        Background data for SHAP. If None, uses 100 samples from X
        
    Returns:
    --------
    str
        HTML content for the SHAP summary plot
    """
    try:
        # Get or create explainer
        explainer = get_explainer(model, background_data)
        
        # Calculate SHAP values with progress indicator
        with st.spinner('Calculating SHAP values for summary plot...'):
            shap_values = explainer.shap_values(X)

            feature_names = np.array(feature_names)
            
            # Create summary plot as HTML
            summary_plot = shap.summary_plot(
                shap_values,  # SHAP values for class 1 (fraud)
                X,
                feature_names=feature_names,
                show=False,
                plot_size=(20, 10)
            )
            return summary_plot
            
    except Exception as e:
        raise Exception(f"An error occurred while generating SHAP summary: {str(e)}")

def explain_with_lime(model, transaction, feature_names, class_names=['Not Fraud', 'Fraud'], 
                     num_features=10, num_samples=5000):
    """
    Generate LIME explanation for a single transaction prediction.
    
    Parameters:
    -----------
    model : sklearn.estimator
        Trained model to explain
    transaction : numpy.ndarray or pandas.Series
        Single transaction to explain
    feature_names : list
        List of feature names
    class_names : list, optional
        Names of the classes. Defaults to ['Not Fraud', 'Fraud']
    num_features : int, optional
        Number of features to show in explanation. Defaults to 10
    num_samples : int, optional
        Number of samples to generate for LIME. Defaults to 5000
        
    Returns:
    --------
    lime.explanation.Explanation
        LIME explanation object
    """
    try:
        # Convert transaction to numpy array if it's a pandas Series
        if isinstance(transaction, pd.Series):
            transaction = transaction.values
            
        # Reshape transaction if needed
        if len(transaction.shape) == 1:
            transaction = transaction.reshape(1, -1)
            
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=transaction,  # We'll use the transaction as reference
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
        
        # Generate explanation
        explanation = explainer.explain_instance(
            data_row=transaction[0],
            predict_fn=model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation
        
    except Exception as e:
        raise Exception(f"An error occurred while generating LIME explanation: {str(e)}")

def plot_lime_explanation(explanation, save_path=None):
    """
    Plot LIME explanation and optionally save it.
    
    Parameters:
    -----------
    explanation : lime.explanation.Explanation
        LIME explanation object
    save_path : str, optional
        Path to save the plot. If None, plot is displayed
    """
    try:
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot explanation
        explanation.as_pyplot_figure()
        plt.title('LIME Explanation for Transaction Prediction')
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        raise Exception(f"An error occurred while plotting LIME explanation: {str(e)}")

def load_models_from_cache(cache_path='models/explainers_cache.pkl'):
    """
    Load models from the explainers cache file.
    
    Parameters:
    -----------
    cache_path : str, optional
        Path to the explainers cache file. Defaults to 'models/explainers_cache.pkl'
        
    Returns:
    --------
    tuple
        (rf_model, nn_model, ensemble_model) - The loaded models
    """
    try:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found at {cache_path}")
            
        with open(cache_path, 'rb') as f:
            explainers = pickle.load(f)
            
        # Extract models from explainers
        rf_model = explainers['random_forest'].model
        nn_model = explainers['neural_network'].model
        ensemble_model = explainers['ensemble'].model
        
        return rf_model, nn_model, ensemble_model
        
    except Exception as e:
        raise Exception(f"Error loading models from cache: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Load pre-built explainers
        models_dir = 'models'
        with open(os.path.join(models_dir, 'explainers_cache.pkl'), 'rb') as f:
            explainers = pickle.load(f)
        
        # Example usage with a sample transaction
        sample_transaction = {
            'step': 1,
            'customer': 1,
            'age': 35,
            'gender': 1,
            'zipcodeOri': 0,
            'merchant': 1,
            'zipMerchant': 0,
            'category': 1,
            'amount': 1
        }
        
        # Load the column mapping
        column_mapping = load_column_mapping()
        print("Column mapping loaded successfully!")
        
        # Preprocess the transaction using the loaded mapping
        processed_transaction = preprocess_single_transaction(sample_transaction, column_mapping)
        
        # Create feature names
        feature_names = list(sample_transaction.keys())
        
        # Display explanations for each model
        model_names = {
            'Random Forest': 'random_forest',
            'Neural Network': 'neural_network',
            'Ensemble': 'ensemble'
        }
        
        for display_name, model_name in model_names.items():
            print(f"\nGenerating SHAP explanation for {display_name}...")
            try:
                explanation = explain_prediction(
                    model_name,
                    processed_transaction,
                    feature_names,
                    explainers
                )
                print("explanation: ", explanation)
                print("type of explanation: ", type(explanation))
                print("len of explanation: ", len(explanation))
                print("explanation.values: ", explanation.values)
                print("explanation.base_values: ", explanation.base_values)
                print("explanation.data: ", explanation.data)
                print("explanation.feature_names: ", explanation.feature_names)
                print("Explanation generated successfully!")
            except Exception as e:
                print(f"Error generating explanation for {display_name}: {str(e)}")
                
    except Exception as e:
        print(f"Error: {str(e)}")