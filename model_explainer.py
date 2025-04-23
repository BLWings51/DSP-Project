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

# Cache for explainers
_explainer_cache = {}

def get_explainer(model, background_data=None):
    """Get or create a cached explainer for the model."""
    model_id = id(model)
    print("model_id: ", model_id)
    
    # First, try to load from memory cache
    if model_id in _explainer_cache:
        print("model_id in _explainer_cache: ", model_id)
        return _explainer_cache[model_id]
    
    # Then, try to load from disk cache
    try:
        cache_path = os.path.join('models', 'explainers_cache.pkl')
        print("cache_path: ", cache_path)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                disk_cache = pickle.load(f)
                print("disk_cache: ", disk_cache)

                # Determine model type
                if isinstance(model, RandomForestClassifier):
                    model_type = 'random_forest'
                elif hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
                    model_type = 'neural_network'
                else:
                    model_type = 'ensemble'
                
                if model_type in disk_cache:
                    explainer = disk_cache[model_type]
                    _explainer_cache[model_id] = explainer
                    return explainer
    except Exception as e:
        st.warning(f"Could not load cached explainer: {str(e)}")
    
    # If no cache available, create new explainer with progress indicator
    with st.spinner('Creating SHAP explainer... This may take a few minutes.'):
        if hasattr(model, 'predict_proba') and not isinstance(model, VotingClassifier):
            # For scikit-learn models (except ensemble)
            explainer = shap.TreeExplainer(model)
        else:
            # For Keras models and ensemble
            if background_data is None:
                background_data = shap.utils.sample(np.zeros((1, model.n_features_in_)), 100)
            explainer = shap.KernelExplainer(
                model.predict_proba,
                background_data,
                output_names=['Not Fraud', 'Fraud']
            )
        _explainer_cache[model_id] = explainer
    
    return _explainer_cache[model_id]

def explain_prediction(model, transaction, feature_names, background_data=None, plot_type='force'):
    """
    Generate SHAP explanation for a single transaction prediction.
    
    Parameters:
    -----------
    model : object
        Trained model to explain
    transaction : numpy.ndarray or pandas.Series
        Single transaction to explain
    feature_names : list
        Names of the features
    background_data : numpy.ndarray, optional
        Background data for SHAP. If None, will use 100 samples from the training data
    plot_type : str, optional
        Type of SHAP plot to generate ('force', 'waterfall', or 'bar'). Defaults to 'force'
        
    Returns:
    --------
    matplotlib.figure.Figure or str
        SHAP explanation plot or HTML content
    """
    try:
        # Ensure transaction is in the correct format
        if isinstance(transaction, pd.Series):
            transaction = transaction.values.reshape(1, -1)
        elif len(transaction.shape) == 1:
            transaction = transaction.reshape(1, -1)
            
        # Get or create explainer
        explainer = get_explainer(model, background_data)
        
        # Calculate SHAP values with progress indicator
        with st.spinner('Calculating SHAP values...'):
            shap_values = explainer.shap_values(transaction)
            expected_value = explainer.expected_value
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (fraud)
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1]
            else:
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[0]

        print("shap_values: ", shap_values)
        
        
        if plot_type == 'force':
            # Ensure shap_values is 2D (n_samples, n_features)
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Check if we have multiple samples
            if len(shap_values.shape) > 1 and shap_values.shape[0] > 1:
                # For multiple samples, use the older SHAP API
                shap.force_plot(
                    expected_value,
                    shap_values,
                    transaction,
                    feature_names=feature_names,
                    matplotlib=False,
                    show=False
                )
            else:
                plt.figure(figsize=(10, 4))
                
                arr_reshaped = np.squeeze(shap_values, axis=0)
                print("arr_reshaped: ", arr_reshaped)
                print("arr_reshaped[:,0]: ", arr_reshaped[:,0])
                print("feature_names: ", feature_names)

                # testArr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            
                # Generate the force plot as HTML
                force_plot = shap.force_plot(
                    expected_value,
                    arr_reshaped[:,0],  # First sample's SHAP values
                    transaction[0],  # First sample's feature values
                    feature_names=feature_names,
                    matplotlib=False
                )
                
                # Convert to HTML and display in Streamlit
                fig = plt.gcf()
                fig.set_size_inches(10, 4)
                plt.tight_layout()
                return fig
            
        elif plot_type == 'waterfall':
            # Create figure for waterfall plot
            fig = plt.figure(figsize=(10, 6))
            
            if len(shap_values.shape) == 3:
                values = shap_values[0, :, 0]
            elif len(shap_values.shape) == 2:
                values = shap_values[0]
            else:
                values = shap_values
                
            explanation = shap.Explanation(
                values=values,
                base_values=expected_value,
                data=transaction[0],
                feature_names=feature_names
            )
            # shap.plots.waterfall(explanation)
            return fig
            
        elif plot_type == 'bar':
            # Create figure for bar plot
            fig = plt.figure(figsize=(10, 6))
            
            if len(shap_values.shape) == 3:
                values = shap_values[0, :, 0]
            elif len(shap_values.shape) == 2:
                values = shap_values[0]
            else:
                values = shap_values
                
            explanation = shap.Explanation(
                values=values,
                base_values=expected_value,
                data=transaction[0],
                feature_names=feature_names
            )
            # shap.plots.bar(explanation)
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
    matplotlib.figure.Figure
        SHAP summary plot
    """
    try:
        # Get or create explainer
        explainer = get_explainer(model, background_data)
        print("explainer: ", explainer)
        
        # Calculate SHAP values with progress indicator
        shap_values = explainer.shap_values(X)
        print("shap_values: ", shap_values)

        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[1],  # SHAP values for class 1 (fraud)
            X,
            feature_names=feature_names,
            show=False
        )
        plt.title('SHAP Summary Plot for Fraud Predictions')
        plt.tight_layout()
        
        return plt.gcf()
            
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

# Example usage
if __name__ == "__main__":
    from data_loader import load_transaction_data, preprocess_data, split_data, train_random_forest
    
    # Load and preprocess data
    df = load_transaction_data()
    df_processed, column_mapping = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Train a Random Forest model
    rf_model = train_random_forest(X_train.values, y_train.values)
    
    # Explain a single transaction
    transaction = X_test.values[0]  # First test transaction
    feature_names = [col for col in X_test.columns if col != 'fraud']
    
    # Generate force plot
    force_plot = explain_prediction(
        rf_model,
        transaction,
        feature_names,
        plot_type='force'
    )
    print("force_plot: ", force_plot)
    force_plot.savefig('force_plot.png')
    plt.close()
    
    # Generate waterfall plot
    waterfall_plot = explain_prediction(
        rf_model,
        transaction,
        feature_names,
        plot_type='waterfall'
    )
    print("waterfall_plot: ", waterfall_plot)
    waterfall_plot.savefig('waterfall_plot.png')
    plt.close()
    
    # Generate global explanation
    summary_plot = explain_model(
        rf_model,
        X_test.values,
        feature_names
    )
    print("summary_plot: ", summary_plot)
    summary_plot.savefig('summary_plot.png')
    plt.close()
    
    # # Explain a single transaction with LIME
    # lime_explanation = explain_with_lime(
    #     rf_model,
    #     transaction,
    #     feature_names,
    #     num_features=10,
    #     num_samples=5000
    # )
    # print("lime_explanation: ", lime_explanation)
    # # Plot and save LIME explanation
    # plot_lime_explanation(lime_explanation, 'lime_explanation.png')
    
    # # Print explanation in text format
    # print("\nLIME Explanation:")
    # print(lime_explanation.as_list())
    
    # # Print prediction probabilities
    # print("\nPrediction Probabilities:")
    # print(f"Not Fraud: {lime_explanation.predict_proba[0]:.4f}")
    # print(f"Fraud: {lime_explanation.predict_proba[1]:.4f}") 