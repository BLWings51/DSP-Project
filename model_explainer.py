import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from preload_explainers import load_background_data
from utils import load_column_mapping, preprocess_single_transaction
import os
import pickle
from utils import load_column_mapping, preprocess_single_transaction, load_model_from_disk

# Cache for explainers
_explainer_cache = {}

def get_explainer(model, background_data=None):
    """
    Return a cached SHAP explainer:
      - TreeExplainer for tree-based models
      - KernelExplainer otherwise
    """
    mid = id(model)
    if mid in _explainer_cache:
        return _explainer_cache[mid]

    if background_data is None:
        background_data = load_background_data(
            n_samples=50,
            n_features=getattr(model, 'n_features_in_', None)
        )

    if isinstance(model, (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier)):
        expl = shap.TreeExplainer(model)
    else:
        predict_fn = getattr(model, 'predict_proba', model)
        expl = shap.KernelExplainer(
            predict_fn,
            background_data,
            output_names=['Not Fraud', 'Fraud']
        )

    _explainer_cache[mid] = expl
    return expl

def explain_prediction(model_name, transaction, feature_names, explainers, plot_type='force'):
    """
    Generate SHAP explanation for a single transaction prediction using pre-built explainers.

    Returns a matplotlib.Figure.
    """
    try:
        # 1) Ensure transaction is 2D float32 array
        if isinstance(transaction, pd.Series):
            arr = transaction.values.astype(np.float32).reshape(1, -1)
        elif isinstance(transaction, np.ndarray):
            arr = transaction.astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        else:
            raise ValueError("Transaction must be a pandas Series or numpy array")

        # 2) Lookup explainer and get SHAP values
        explainer = explainers[model_name]
        raw_shap = explainer(arr)

        # 3) Extract raw values and base value
        if hasattr(raw_shap, "values"):
            vals = raw_shap.values
            exp_val = raw_shap.base_values
        else:
            vals     = raw_shap
            exp_val  = explainer.expected_value

        # 4) Collapse to positive‐class shap values
        #    Handles various shapes: list, (1,n,2), (1,n,1), (1,n)
        if isinstance(vals, list):
            so = np.array(vals[1])
        else:
            vals = np.array(vals)
            if vals.ndim == 3 and vals.shape[2] > 1:
                so = vals[:, :, 1]
            elif vals.ndim == 3:
                so = vals[:, :, 0]
            elif vals.ndim == 2:
                so = vals
            else:
                so = vals.reshape(vals.shape[0], -1)
        single_output = so.reshape(1, -1)

        # 5) Extract scalar base value (positive class)
        ev = np.array(exp_val).flatten()
        base_value = float(ev[-1])

        # 6) Flatten for plotting
        sv = single_output.flatten()
        fv = arr.flatten()

        # === ISSUE #4 FIX: convert to percentage & annotate ===
        sv = sv * 100.0
        bv = base_value * 100.0
        title = (
            f"{model_name.replace('_',' ').title()} SHAP {plot_type.title()} Plot\n"
            f"Base fraud probability = {bv:.1f}%"
        )

        # 7) Plot
        if plot_type == 'force':
            fig = shap.plots.force(
                bv, sv,
                feature_names=feature_names,
                features=fv,
                show=False,
                matplotlib=True
            )
            plt.title(title)
            return fig

        elif plot_type == 'waterfall':
            explanation = shap.Explanation(
                values=sv,
                base_values=bv,
                data=fv,
                feature_names=feature_names
            )
            plt.figure(figsize=(8,6))
            shap.plots.waterfall(
                explanation,
                max_display=len(sv),
                show=False
            )
            plt.title(title)
            plt.tight_layout()
            return plt.gcf()

        elif plot_type == 'bar':
            explanation = shap.Explanation(
                values=sv,
                base_values=bv,
                data=fv,
                feature_names=feature_names
            )
            plt.figure(figsize=(8,6))
            shap.plots.bar(
                explanation,
                max_display=len(sv),
                show=False
            )
            plt.title(title)
            plt.tight_layout()
            return plt.gcf()

        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    except Exception as e:
        raise Exception(f"An error occurred while generating SHAP explanation: {e}")

def explain_with_lime(model, transaction, feature_names, class_names=['Not Fraud','Fraud'],
                     num_features=10, num_samples=5000):
    """
    Generate a LIME explanation for a single transaction.
    """
    from lime.lime_tabular import LimeTabularExplainer

    if isinstance(transaction, pd.Series):
        arr = transaction.values
    else:
        arr = np.asarray(transaction).reshape(-1)

    explainer = LimeTabularExplainer(
        training_data=arr.reshape(1,-1),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=arr,
        predict_fn=model.predict_proba,
        num_features=num_features,
        num_samples=num_samples
    )
    return explanation


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

def load_explainers_from_cache(cache_path='models/explainers_cache.pkl'):
    """
    Loads the dict of SHAP explainers under keys
      'random_forest', 'neural_network', 'voting', 'adaboost', 'stacking'
    from disk.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Explainer cache not found at {cache_path}")
    with open(cache_path, 'rb') as f:
        explainers = pickle.load(f)

    required = ['random_forest','neural_network','voting','adaboost','stacking']
    missing  = [k for k in required if k not in explainers]
    if missing:
        raise KeyError(f"Cache is missing explainers for: {missing}")

    return explainers

# Example usage
if __name__ == "__main__":
    try:
        # 1) Where our models live
        models_dir = 'models'
        
        # 2) Load the column mapping (for single‐row preprocessing)
        column_mapping = load_column_mapping()
        print("Loaded column mapping.")

        # 3) Paths for each saved model
        paths = {
            'random_forest':  os.path.join(models_dir, 'rf_model.joblib'),
            'neural_network': os.path.join(models_dir, 'nn_model.h5'),
            'voting':         os.path.join(models_dir, 'ensemble_voting.joblib'),
            'adaboost':       os.path.join(models_dir, 'ensemble_adaboost.joblib'),
            'stacking':       os.path.join(models_dir, 'ensemble_stacking.joblib'),
        }

        # 4) Load each model from disk
        models = {}
        models['random_forest']  = load_model_from_disk(paths['random_forest'],  'rf')
        models['neural_network'] = load_model_from_disk(paths['neural_network'], 'nn')
        models['voting']         = load_model_from_disk(paths['voting'],         'ensemble')
        models['adaboost']       = load_model_from_disk(paths['adaboost'],       'ensemble')
        models['stacking']       = load_model_from_disk(paths['stacking'],       'ensemble')
        print("Models loaded:", list(models.keys()))

        # 5) One background dataset for all explainers
        background_data = load_background_data(
            n_samples=100,
            n_features=getattr(models['random_forest'], 'n_features_in_', None)
        )

        # 6) Build a SHAP explainer for each
        explainers = {
            name: get_explainer(mdl, background_data)
            for name, mdl in models.items()
        }
        print("SHAP explainers ready.")

        # 7) Define a raw sample transaction
        sample_transaction = {
            'step':       1,
            'customer':   'C123456',
            'age':        35,
            'gender':     'M',
            'zipcodeOri': '28007',
            'merchant':   'M654321',
            'zipMerchant':'28007',
            'category':   'es_transportation',
            'amount':     42.50
        }

        # 8) Preprocess into numeric vector
        processed = preprocess_single_transaction(sample_transaction, column_mapping)
        feature_names = list(sample_transaction.keys())

        # 9) Generate and show a SHAP force plot for each model
        for name in ['random_forest', 'neural_network', 'voting', 'adaboost', 'stacking']:
            print(f"\n→ Explaining {name} …")
            fig = explain_prediction(
                model_name   = name,
                transaction  = processed,
                feature_names= feature_names,
                explainers   = explainers,
                plot_type    = 'force'
            )
            plt.show()

    except Exception as e:
        print("Error in explanation main:", e)
        import traceback; traceback.print_exc()
