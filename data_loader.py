import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from preload_explainers import preload_explainers
import joblib
import os
import pickle
from utils import save_column_mapping, save_model, load_model_from_disk, KerasBinaryClassifier

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


def train_adaboost(X_train, y_train, base_estimator=None, n_estimators=100, learning_rate=1.0, random_state=42):
    """
    Train an AdaBoostClassifier on your data.
    """
    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    ada = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    ada.fit(X_train, y_train)
    return ada

def train_stacking(X_train, y_train, estimators=None, final_estimator=None, cv=5):
    """
    Train a StackingClassifier on your data.
    """
    if estimators is None:
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42)),
        ]
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )
    stack.fit(X_train, y_train)
    return stack

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



def create_ensemble(X_train, y_train, voting='soft', kind='voting'):
    """
    Create and train an ensemble model.
    kind: 'voting' | 'adaboost' | 'stacking'
    """
    try:
        # Always train the NN + wrap
        nn_model, _ = train_neural_network(X_train, y_train)
        nn_classifier = KerasBinaryClassifier(
            model=nn_model,
            n_features=X_train.shape[1]
        )

        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        if kind == 'voting':
            # your original voting classifier
            ensemble = VotingClassifier(
                estimators=[('nn', nn_classifier), ('rf', rf_classifier)],
                voting=voting,
                n_jobs=-1
            )

        elif kind == 'adaboost':
            ada = train_adaboost(X_train, y_train)
            ensemble = VotingClassifier(
                estimators=[('nn', nn_classifier), ('ada', ada)],
                voting=voting,
                n_jobs=-1
            )

        elif kind == 'stacking':
            stack = train_stacking(X_train, y_train)
            ensemble = VotingClassifier(
                estimators=[('nn', nn_classifier), ('stack', stack)],
                voting=voting,
                n_jobs=-1
            )

        else:
            raise ValueError(f"Unknown ensemble kind: {kind}")

        ensemble.fit(X_train, y_train)
        return ensemble

    except Exception as e:
        raise Exception(f"An error occurred while creating the ensemble: {str(e)}")

# Example usage:
if __name__ == "__main__":
    try:
        # 1) Load & preprocess
        df = load_transaction_data()
        print("Data loaded:", df.shape)
        df_processed, column_mapping = preprocess_data(df)
        save_column_mapping(column_mapping)
        print("Data preprocessed; mapping saved.")

        # 2) Split
        X_train, X_test, y_train, y_test = split_data(df_processed)
        X_train_values = X_train.values
        X_test_values  = X_test.values
        y_train_values = y_train.values
        y_test_values  = y_test.values
        print("Split: train =", X_train.shape, "test =", X_test.shape)

        # 3) Ensure models directory
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # 4) Train or load RF
        rf_path = os.path.join(models_dir, "rf_model.joblib")
        if os.path.exists(rf_path):
            rf_model = load_model_from_disk(rf_path, "rf")
            print("RF loaded from disk.")
        else:
            rf_model = train_random_forest(X_train_values, y_train_values)
            save_model(rf_model, rf_path, "rf")
            print("RF trained and saved.")
        evaluate_model(rf_model, X_test_values, y_test_values)

        # 5) Train or load NN
        nn_path = os.path.join(models_dir, "nn_model.h5")
        if os.path.exists(nn_path):
            nn_model = load_model_from_disk(nn_path, "nn")
            print("NN loaded from disk.")
        else:
            nn_model, _ = train_neural_network(X_train_values, y_train_values)
            save_model(nn_model, nn_path, "nn")
            print("NN trained and saved.")
        evaluate_model(nn_model, X_test_values, y_test_values)

        # 6) Train & evaluate each ensemble kind
        for kind in ("voting", "adaboost", "stacking"):
            path = os.path.join(models_dir, f"ensemble_{kind}.joblib")
            if os.path.exists(path):
                ens = load_model_from_disk(path, "ensemble")
                print(f"{kind.title()} ensemble loaded.")
            else:
                ens = create_ensemble(
                    X_train_values, y_train_values,
                    voting="soft",
                    kind=kind
                )
                save_model(ens, path, "ensemble")
                print(f"{kind.title()} ensemble trained and saved.")
            print(f"\nEvaluating {kind.title()} Ensemble:")
            evaluate_model(ens, X_test_values, y_test_values)

        # 7) Preload SHAP/LIME explainers
        print("\nPreloading explainers...")
        explainers = preload_explainers(models_dir=models_dir, background_samples=100)
        print("✅ Explainers cached!")
        
    except Exception as e:
        print("Fatal error in main:", str(e))
        import traceback; traceback.print_exc()
