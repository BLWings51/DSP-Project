import os
import pickle
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, ClassifierMixin

def save_column_mapping(column_mapping, file_path='models/column_mapping.pkl'):
    """
    Save the column mapping to a file.
    
    Parameters:
    -----------
    column_mapping : dict
        Dictionary containing the mapping of categorical values to encoded values
        and any other preprocessing information
    file_path : str, optional
        Path to save the column mapping. Defaults to 'models/column_mapping.pkl'
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the mapping
        with open(file_path, 'wb') as f:
            pickle.dump(column_mapping, f)
            
    except Exception as e:
        raise Exception(f"Error saving column mapping: {str(e)}")

def load_column_mapping(file_path='models/column_mapping.pkl'):
    """
    Load the column mapping from a file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the column mapping file. Defaults to 'models/column_mapping.pkl'
        
    Returns:
    --------
    dict
        Dictionary containing the mapping of categorical values to encoded values
        and any other preprocessing information
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Column mapping file not found at {file_path}")
            
        with open(file_path, 'rb') as f:
            column_mapping = pickle.load(f)
            
        return column_mapping
        
    except Exception as e:
        raise Exception(f"Error loading column mapping: {str(e)}")

def preprocess_single_transaction(transaction, column_mapping):
    """
    Preprocess a single transaction using the same logic as preprocess_data.
    
    Parameters:
    -----------
    transaction : dict or pandas.Series
        Single transaction data with feature values
    column_mapping : dict
        Dictionary containing the mapping of categorical values to encoded values
        and any other preprocessing information (e.g., median values)
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed transaction as a numpy array
    """
    try:
        # Convert to pandas Series if it's a dictionary
        if isinstance(transaction, dict):
            transaction = pd.Series(transaction)
            
        # Create a copy to avoid modifying the original
        transaction_processed = transaction.copy()
        
        # Handle age field if present
        if 'age' in transaction_processed:
            # Convert age to string to handle mixed types
            age_str = str(transaction_processed['age'])
            
            # Replace 'U' and any quoted 'U' with median age
            if age_str.replace("'", "") == 'U':
                transaction_processed['age'] = column_mapping.get('age_median', 35)  # Default to 35 if median not found
            else:
                # Convert to numeric
                transaction_processed['age'] = pd.to_numeric(age_str, errors='coerce')
                if pd.isna(transaction_processed['age']):
                    transaction_processed['age'] = column_mapping.get('age_median', 35)
        
        # Encode categorical variables
        for col, mapping in column_mapping.items():
            if col != 'age_median' and col in transaction_processed:
                # Get the original value
                original_value = transaction_processed[col]
                
                # Map to encoded value
                if original_value in mapping:
                    transaction_processed[col] = mapping[original_value]
                else:
                    # If value not in mapping, use a default value (e.g., 0)
                    transaction_processed[col] = 0
        
        # Convert to numpy array and ensure correct shape
        transaction_array = transaction_processed.values.reshape(1, -1)
        
        return transaction_array
        
    except Exception as e:
        raise Exception(f"Error preprocessing transaction: {str(e)}")

def save_model(model, model_path, model_type='ensemble'):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : object
        Trained model to save (scikit-learn or Keras)
    model_path : str
        Path where to save the model
    model_type : str, optional
        Type of model ('ensemble', 'rf', or 'nn'). Defaults to 'ensemble'
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if model_type == 'nn':
            # Save Keras model
            model.save(model_path)
            print(f"Keras model saved to {model_path}")
        else:
            # Save scikit-learn model
            joblib.dump(model, model_path)
            print(f"Scikit-learn model saved to {model_path}")
            
    except Exception as e:
        raise Exception(f"An error occurred while saving the model: {str(e)}")

def load_model_from_disk(model_path, model_type='ensemble'):
    """
    Load a saved model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    model_type : str, optional
        Type of model ('ensemble', 'rf', or 'nn'). Defaults to 'ensemble'
        
    Returns:
    --------
    object
        Loaded model
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        if model_type == 'nn':
            # Load Keras model
            model = load_model(model_path)
            print(f"Keras model loaded from {model_path}")
        else:
            # Load scikit-learn model
            model = joblib.load(model_path)
            print(f"Scikit-learn model loaded from {model_path}")
            
        return model
        
    except Exception as e:
        raise Exception(f"An error occurred while loading the model: {str(e)}") 
    
class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper for Keras binary classifier with proper scikit-learn interface."""
    
    def __init__(self, model=None, n_features=None):
        self.model = model
        self.n_features = n_features  # Changed from _n_features_in to n_features
        self._n_features_in = n_features  # Keep this for scikit-learn compatibility
        
    def fit(self, X, y=None):
        """Dummy fit method that just records feature count."""
        self._n_features_in = X.shape[1]
        self.n_features = X.shape[1]  # Update both attributes
        return self  # Always return self
        
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not initialized")
        # Get raw predictions and ensure they're 1D
        p = self.model.predict(X, verbose=0).reshape(-1)
        # Stack the "not fraud" and "fraud" columns
        return np.column_stack([1 - p, p])
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'model': self.model,
            'n_features': self.n_features
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    @property 
    def n_features_in_(self):
        """Number of features seen during fit."""
        return self._n_features_in