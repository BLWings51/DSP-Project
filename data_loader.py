import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import joblib
import os
import pickle
from utils import save_column_mapping, save_model, load_model_from_disk

def load_transaction_data(file_path='banksimData.csv'):
    """
    Load transaction data from a CSV file into a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file. Defaults to 'banksimData.csv'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the transaction data with columns:
        - step: Transaction step
        - customer: Customer identifier
        - age: Customer age
        - gender: Customer gender
        - merchant: Merchant identifier
        - category: Transaction category
        - amount: Transaction amount
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Verify that the required columns exist
        required_columns = ['step', 'customer', 'age', 'gender', 'merchant', 'category', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {str(e)}")
    

def preprocess_data(df, fraud_column='fraud'):
    """
    Preprocess the transaction data by encoding categorical variables and normalizing numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing transaction data
    fraud_column : str, optional
        Name of the fraud indicator column. Defaults to 'fraud'
        
    Returns:
    --------
    tuple
        (processed_df, column_mapping)
        processed_df: Preprocessed DataFrame
        column_mapping: Dictionary mapping original categorical values to encoded values
    """
    try:
        # Create a copy to avoid modifying the original DataFrame
        df_processed = df.copy()
        
        # Initialize column mapping dictionary
        column_mapping = {}
        
        # Handle age field with 'U' values and single quotes
        if 'age' in df_processed.columns:
            # First convert all age values to strings to handle mixed types
            df_processed['age'] = df_processed['age'].astype(str)
            
            # Replace 'U' and any quoted 'U' with NaN
            df_processed['age'] = df_processed['age'].str.replace("'", "").replace('U', np.nan)
            
            # Convert to numeric (non-numeric becomes NaN)
            df_processed['age'] = pd.to_numeric(df_processed['age'], errors='coerce')
            
            # Calculate median age excluding NaN values
            median_age = df_processed['age'].median()
            
            # Replace NaN values with median age
            df_processed['age'] = df_processed['age'].fillna(median_age)
            
            # Store the median age in column mapping for reference
            column_mapping['age_median'] = median_age
        
        # Identify categorical columns (excluding the fraud column)
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != fraud_column]
        
        # Encode categorical variables
        for col in categorical_columns:
            # Create mapping from original values to encoded values
            unique_values = df_processed[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            column_mapping[col] = mapping
            
            # Apply encoding
            df_processed[col] = df_processed[col].map(mapping)
        
        # Normalize numerical features (excluding the fraud column)
        numerical_columns = df_processed.select_dtypes(include=['float64', 'int64']).columns
        numerical_columns = [col for col in numerical_columns if col != fraud_column]
        
        for col in numerical_columns:
            # Skip if all values are the same
            if df_processed[col].nunique() > 1:
                df_processed[col] = (df_processed[col] - df_processed[col].min()) / \
                                  (df_processed[col].max() - df_processed[col].min())
        
        return df_processed, column_mapping
        
    except Exception as e:
        raise Exception(f"Error preprocessing data: {str(e)}")

def split_data(df, target_column='fraud', test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and test sets for machine learning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the features and target variable
    target_column : str, optional
        Name of the target column. Defaults to 'fraud'
    test_size : float, optional
        Proportion of the dataset to include in the test split. Defaults to 0.2
    random_state : int, optional
        Controls the shuffling applied to the data before splitting. Defaults to 42
        
    Returns:
    --------
    tuple
        A tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training target
        - y_test: Test target
    """
    try:
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Preserve the proportion of target classes
        )
        
        return X_train, X_test, y_train, y_test
        
    except KeyError:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame")
    except Exception as e:
        raise Exception(f"An error occurred while splitting the data: {str(e)}")

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest classifier with balanced class weights.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training target
    n_estimators : int, optional
        Number of trees in the forest. Defaults to 100
    max_depth : int, optional
        Maximum depth of the tree. Defaults to None (unlimited)
    random_state : int, optional
        Controls the randomness of the estimator. Defaults to 42
        
    Returns:
    --------
    RandomForestClassifier
        Trained Random Forest model
    """
    try:
        # Initialize the Random Forest classifier
        rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',  # Handle class imbalance
            random_state=random_state,
            n_jobs=-1  # Use all available processors
        )
        
        # Train the model
        rf_classifier.fit(X_train, y_train)
        
        return rf_classifier
        
    except Exception as e:
        raise Exception(f"An error occurred while training the Random Forest: {str(e)}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print performance metrics.
    
    Parameters:
    -----------
    model : sklearn.estimator or keras.Model
        Trained model to evaluate
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test target
        
    Returns:
    --------
    dict
        Dictionary containing the evaluation metrics
    """
    # Make predictions based on model type
    if hasattr(model, 'predict_proba'):
        # For scikit-learn models and ensemble
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        # For binary classification, get probabilities for positive class
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
    else:
        # For Keras models
        y_pred_proba = model.predict(X_test, verbose=0)
        
        # Handle both single-output and multi-output models
        if len(y_pred_proba.shape) == 1:
            # Single output (binary classification)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Multi-output (multi-class classification)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba.flatten()
    
    # Ensure predictions are in the correct shape
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

def train_neural_network(X_train, y_train, X_val=None, y_val=None, 
                        hidden_units=[64, 32], dropout_rate=0.3,
                        learning_rate=0.001, batch_size=32, epochs=50,
                        validation_split=0.2):
    """
    Create and train a neural network for binary classification.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training target
    X_val : numpy.ndarray, optional
        Validation features. If None, will use validation_split
    y_val : numpy.ndarray, optional
        Validation target. If None, will use validation_split
    hidden_units : list, optional
        Number of units in each hidden layer. Defaults to [64, 32]
    dropout_rate : float, optional
        Dropout rate for regularization. Defaults to 0.3
    learning_rate : float, optional
        Learning rate for the optimizer. Defaults to 0.001
    batch_size : int, optional
        Batch size for training. Defaults to 32
    epochs : int, optional
        Number of training epochs. Defaults to 50
    validation_split : float, optional
        Proportion of training data to use for validation. Defaults to 0.2
        
    Returns:
    --------
    keras.Model
        Trained neural network model
    """
    try:
        # Create the model
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_units[0], activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_units[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        if X_val is not None and y_val is not None:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=1
            )
        else:
            history = model.fit(
                X_train, y_train,
                validation_split=validation_split,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=1
            )
        
        return model, history
        
    except Exception as e:
        raise Exception(f"An error occurred while training the neural network: {str(e)}")

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

def create_ensemble(X_train, y_train, voting='soft'):
    """
    Create and train an ensemble model combining Random Forest and Neural Network.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training target
    voting : str, optional
        Voting strategy ('hard' or 'soft'). Defaults to 'soft'
        
    Returns:
    --------
    VotingClassifier
        Trained ensemble model
    """
    try:
        # Create and train the neural network model first
        nn_model, _ = train_neural_network(X_train, y_train)
        
        # Create the neural network classifier wrapper
        nn_classifier = KerasBinaryClassifier(
            model=nn_model,
            n_features=X_train.shape[1]  # This is now the correct parameter name
        )
        
        # Create the random forest classifier
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        
        # Create the ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('nn', nn_classifier),
                ('rf', rf_classifier)
            ],
            voting=voting,
            n_jobs=-1
        )
        
        # Train the ensemble
        ensemble.fit(X_train, y_train)
        
        return ensemble
        
    except Exception as e:
        raise Exception(f"An error occurred while creating the ensemble: {str(e)}")

# Example usage:
if __name__ == "__main__":
    try:
        # Load the data
        df = load_transaction_data()
        print("Data loaded successfully!")
        print(f"Number of rows: {len(df)}")
        print(f"Original columns: {df.columns.tolist()}")
        
        # Preprocess the data
        df_processed, column_mapping = preprocess_data(df)
        print("\nPreprocessed data:")
        print(f"Number of rows: {len(df_processed)}")
        print(f"Processed columns: {df_processed.columns.tolist()}")
        
        # Save the column mapping
        save_column_mapping(column_mapping)
        print("\nColumn mapping saved successfully!")
        
        # Split the data
        X_train, X_test, y_train, y_test = split_data(df_processed)
        print("\nData split results:")
        print(f"Training set size: {len(X_train)} samples")
        print(f"Test set size: {len(X_test)} samples")
        print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")
        
        # Convert data to numpy arrays if they aren't already
        X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
        y_test_values = y_test.values if isinstance(y_test, pd.Series) else y_test
        
        # Check for existing models
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Paths for model files
        ensemble_path = os.path.join(models_dir, 'ensemble_model.joblib')
        nn_path = os.path.join(models_dir, 'nn_model.h5')
        rf_path = os.path.join(models_dir, 'rf_model.joblib')
        
        # Initialize model variables
        rf_model = None
        nn_model = None
        ensemble_model = None
        
        # Try to load existing models
        try:
            print("\nAttempting to load existing models...")
            if os.path.exists(rf_path):
                rf_model = load_model_from_disk(rf_path, 'rf')
                print("Random Forest model loaded successfully!")
                
                # Evaluate Random Forest
                print("\nEvaluating Random Forest model...")
                rf_metrics = evaluate_model(rf_model, X_test_values, y_test_values)
                print("\nRandom Forest Metrics:")
                for metric, value in rf_metrics.items():
                    print(f"{metric}: {value:.4f}")
            
            if os.path.exists(nn_path):
                nn_model = load_model_from_disk(nn_path, 'nn')
                print("Neural Network model loaded successfully!")
                
                # Evaluate Neural Network
                print("\nEvaluating Neural Network model...")
                nn_metrics = evaluate_model(nn_model, X_test_values, y_test_values)
                print("\nNeural Network Metrics:")
                for metric, value in nn_metrics.items():
                    print(f"{metric}: {value:.4f}")
            
            if os.path.exists(ensemble_path):
                ensemble_model = load_model_from_disk(ensemble_path, 'ensemble')
                print("Ensemble model loaded successfully!")
                
                # Evaluate Ensemble
                print("\nEvaluating Ensemble model...")
                ensemble_metrics = evaluate_model(ensemble_model, X_test_values, y_test_values)
                print("\nEnsemble Metrics:")
                for metric, value in ensemble_metrics.items():
                    print(f"{metric}: {value:.4f}")
                    
        except Exception as e:
            print(f"\nError loading models: {str(e)}")
            print("Will train new models...")
        
        # Train models if they don't exist or failed to load
        if rf_model is None:
            print("\nTraining Random Forest model...")
            rf_model = train_random_forest(X_train_values, y_train_values)
            save_model(rf_model, rf_path, 'rf')
            print("Random Forest model saved!")
            
            # Evaluate Random Forest
            print("\nEvaluating Random Forest model...")
            rf_metrics = evaluate_model(rf_model, X_test_values, y_test_values)
            print("\nRandom Forest Metrics:")
            for metric, value in rf_metrics.items():
                print(f"{metric}: {value:.4f}")
        
        if nn_model is None:
            print("\nTraining Neural Network model...")
            nn_model, history = train_neural_network(X_train_values, y_train_values)
            save_model(nn_model, nn_path, 'nn')
            print("Neural Network model saved!")
            
            # Evaluate Neural Network
            print("\nEvaluating Neural Network model...")
            nn_metrics = evaluate_model(nn_model, X_test_values, y_test_values)
            print("\nNeural Network Metrics:")
            for metric, value in nn_metrics.items():
                print(f"{metric}: {value:.4f}")
        
        if ensemble_model is None:
            print("\nTraining Ensemble model...")
            ensemble_model = create_ensemble(X_train_values, y_train_values)
            save_model(ensemble_model, ensemble_path, 'ensemble')
            print("Ensemble model saved!")
            
            # Evaluate Ensemble
            print("\nEvaluating Ensemble model...")
            ensemble_metrics = evaluate_model(ensemble_model, X_test_values, y_test_values)
            print("\nEnsemble Metrics:")
            for metric, value in ensemble_metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Preload explainers after all models are trained/loaded
        print("\nPreloading explainers...")
        from preload_explainers import preload_explainers
        explainers = preload_explainers()
        if explainers:
            print("✅ Successfully preloaded all explainers!")
        else:
            print("❌ Failed to preload explainers.")
        
        print("\nAll models, column mapping, and explainers have been saved successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 