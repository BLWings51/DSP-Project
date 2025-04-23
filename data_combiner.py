import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
from data_loader import load_transaction_data, preprocess_data

def combine_datasets(original_data_path='banksimData.csv', 
                    synthetic_data_path='data/synthetic_fraud.csv',
                    output_path='data/combined_data.csv',
                    fraud_column='fraud',
                    random_state=42):
    """
    Combine original and synthetic fraud data, ensuring proper label alignment and shuffling.
    
    Parameters:
    -----------
    original_data_path : str
        Path to the original dataset
    synthetic_data_path : str
        Path to the synthetic fraud dataset
    output_path : str
        Path to save the combined dataset
    fraud_column : str
        Name of the fraud indicator column
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Combined and shuffled dataset
    """
    try:
        # Load original data
        print("Loading original data...")
        original_df = load_transaction_data(original_data_path)
        
        # Load synthetic data
        print("Loading synthetic data...")
        if not os.path.exists(synthetic_data_path):
            raise FileNotFoundError(f"Synthetic data file not found at {synthetic_data_path}")
        synthetic_df = pd.read_csv(synthetic_data_path)
        
        # Verify column consistency
        if not set(original_df.columns) == set(synthetic_df.columns):
            raise ValueError("Original and synthetic datasets have different columns")
        
        # Ensure fraud column exists
        if fraud_column not in original_df.columns or fraud_column not in synthetic_df.columns:
            raise ValueError(f"Fraud column '{fraud_column}' not found in datasets")
        
        # Combine datasets
        print("Combining datasets...")
        combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        
        # Shuffle the combined dataset
        print("Shuffling combined dataset...")
        combined_df = shuffle(combined_df, random_state=random_state)
        
        # Save the combined dataset
        print(f"Saving combined dataset to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Original dataset size: {len(original_df)}")
        print(f"Synthetic fraud samples: {len(synthetic_df)}")
        print(f"Combined dataset size: {len(combined_df)}")
        print("\nFraud distribution:")
        print(combined_df[fraud_column].value_counts(normalize=True).round(4))
        
        return combined_df
        
    except Exception as e:
        raise Exception(f"Error combining datasets: {str(e)}")

def verify_data_quality(combined_df, fraud_column='fraud'):
    """
    Verify the quality of the combined dataset.
    
    Parameters:
    -----------
    combined_df : pandas.DataFrame
        Combined dataset to verify
    fraud_column : str
        Name of the fraud indicator column
    """
    try:
        print("\nVerifying data quality...")
        
        # Check for missing values
        missing_values = combined_df.isnull().sum()
        if missing_values.any():
            print("\nMissing values found:")
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found")
        
        # Check data types
        print("\nData types:")
        print(combined_df.dtypes)
        
        # Check value ranges for numerical columns
        numerical_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
        print("\nNumerical columns ranges:")
        for col in numerical_cols:
            if col != fraud_column:
                print(f"{col}: [{combined_df[col].min():.2f}, {combined_df[col].max():.2f}]")
        
        # Check for duplicate rows
        duplicates = combined_df.duplicated().sum()
        if duplicates > 0:
            print(f"\nWarning: Found {duplicates} duplicate rows")
        else:
            print("No duplicate rows found")
            
    except Exception as e:
        raise Exception(f"Error verifying data quality: {str(e)}")

if __name__ == "__main__":
    # Combine datasets
    combined_df = combine_datasets(
        original_data_path='banksimData.csv',
        synthetic_data_path='data/synthetic_fraud.csv',
        output_path='data/combined_data.csv'
    )
    
    # Verify data quality
    verify_data_quality(combined_df)
    
    print("\nDataset combination and verification complete!") 