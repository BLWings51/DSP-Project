import shap
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
import os
from data_loader import KerasBinaryClassifier

def preload_explainers(models_dir='models', background_samples=100):
    """Preload explainers with robust feature count handling."""
    explainers = {}
    
    try:
        # 1. Load all models
        rf_model = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
        keras_model = load_model(os.path.join(models_dir, 'nn_model.h5'))
        ensemble_model = joblib.load(os.path.join(models_dir, 'ensemble_model.joblib'))
        
        # 2. Get feature count from RandomForest
        n_features = rf_model.n_features_in_
        
        # 3. Initialize Keras wrapper
        nn_model = KerasBinaryClassifier(model=keras_model, n_features=n_features)
        
        # 4. Prepare models dictionary
        models = {
            'random_forest': rf_model,
            'neural_network': nn_model,
            'ensemble': ensemble_model
        }
        
        # 5. Special handling for VotingClassifier
        if isinstance(ensemble_model, VotingClassifier):
            # Create named estimators list for proper handling
            estimators = []
            for name, est in ensemble_model.named_estimators_.items():
                if isinstance(est, KerasBinaryClassifier):
                    # Clone and initialize Keras wrapper
                    new_est = KerasBinaryClassifier(
                        model=est.model,
                        n_features=n_features
                    )
                    new_est.fit(np.zeros((1, n_features)))  # Dummy fit
                else:
                    # For sklearn estimators
                    est.fit(np.zeros((1, n_features)), [0])  # Dummy fit
                estimators.append((name, est))
            
            # Recreate VotingClassifier with initialized estimators
            ensemble_model.estimators_ = [est for _, est in estimators]
            ensemble_model.named_estimators_ = {name: est for name, est in estimators}
            ensemble_model._n_features_in = n_features
        
        # 6. Create explainers
        for name, model in models.items():
            if isinstance(model, RandomForestClassifier):
                explainers[name] = shap.TreeExplainer(model)
            else:
                background = shap.utils.sample(np.zeros((1, n_features)), background_samples)
                explainers[name] = shap.KernelExplainer(
                    model.predict_proba, 
                    background,
                    output_names=['Not Fraud', 'Fraud']
                )
        
        # 7. Save explainers
        with open(os.path.join(models_dir, 'explainers_cache.pkl'), 'wb') as f:
            pickle.dump(explainers, f)
            
        return explainers
        
    except Exception as e:
        print(f"Error during explainer preloading: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    
    explainers = preload_explainers()
    if explainers:
        print("✅ Successfully preloaded all explainers!")
    else:
        print("❌ Failed to preload explainers.")