"""
Data loading utilities for the drug classification project.

This module handles loading the raw data and basic data validation.
Think of this as the "data reader" that brings our CSV file into Python.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to the path so we can import our utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.helpers import print_dataset_info, calculate_class_distribution

class DataLoader:
    """
    A class to handle loading and basic validation of the drug dataset.
    
    This class makes it easy to load our data and check that everything looks correct.
    """
    
    def __init__(self, data_path="data/raw/drug200.csv"):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (str): Path to the raw data CSV file
        """
        self.data_path = data_path
        self.raw_data = None
    
    def load_data(self):
        """
        Load the drug dataset from CSV file.
        
        Returns:
            pandas.DataFrame: The loaded dataset
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            Exception: If there's an error loading the data
        """
        try:
            # Check if file exists
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load the CSV file
            print(f"Loading data from: {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
            
            # Basic validation
            self._validate_data()
            
            print("✅ Data loaded successfully!")
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def _validate_data(self):
        """
        Perform basic validation on the loaded data.
        
        This checks that our data has the expected structure and content.
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Expected columns for the drug dataset
        expected_columns = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
        
        # Check if all expected columns are present
        missing_columns = set(expected_columns) - set(self.raw_data.columns)
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")
        
        # Check for empty dataset
        if len(self.raw_data) == 0:
            raise ValueError("Dataset is empty!")
        
        # Check data types
        self._validate_data_types()
        
        print("✅ Data validation passed!")
    
    def _validate_data_types(self):
        """
        Validate that columns have expected data types.
        """
        # Age should be numeric
        if not pd.api.types.is_numeric_dtype(self.raw_data['Age']):
            print("⚠️  Warning: Age column is not numeric")
        
        # Na_to_K should be numeric
        if not pd.api.types.is_numeric_dtype(self.raw_data['Na_to_K']):
            print("⚠️  Warning: Na_to_K column is not numeric")
        
        # Categorical columns should be strings/objects
        categorical_cols = ['Sex', 'BP', 'Cholesterol', 'Drug']
        for col in categorical_cols:
            if pd.api.types.is_numeric_dtype(self.raw_data[col]):
                print(f"⚠️  Warning: {col} column appears to be numeric but should be categorical")
    
    def get_data_summary(self):
        """
        Get a comprehensive summary of the loaded data.
        
        Returns:
            dict: Summary statistics and information about the dataset
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Print detailed information
        print_dataset_info(self.raw_data, "DRUG CLASSIFICATION DATASET")
        
        # Show class distribution
        calculate_class_distribution(self.raw_data['Drug'])
        
        # Create summary dictionary
        summary = {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'data_types': self.raw_data.dtypes.to_dict(),
            'target_classes': self.raw_data['Drug'].unique().tolist(),
            'class_counts': self.raw_data['Drug'].value_counts().to_dict()
        }
        
        return summary
    
    def get_feature_info(self):
        """
        Get detailed information about each feature in the dataset.
        
        Returns:
            dict: Information about each feature
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        feature_info = {}
        
        for column in self.raw_data.columns:
            if column == 'Drug':  # Skip target variable
                continue
                
            col_data = self.raw_data[column]
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Numeric feature
                feature_info[column] = {
                    'type': 'numeric',
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'unique_values': col_data.nunique()
                }
            else:
                # Categorical feature
                feature_info[column] = {
                    'type': 'categorical',
                    'unique_values': col_data.nunique(),
                    'categories': col_data.unique().tolist(),
                    'value_counts': col_data.value_counts().to_dict()
                }
        
        # Print feature information
        print("\n" + "="*60)
        print("FEATURE INFORMATION")
        print("="*60)
        
        for feature, info in feature_info.items():
            print(f"\n📊 {feature.upper()}:")
            print(f"   Type: {info['type']}")
            
            if info['type'] == 'numeric':
                print(f"   Range: {info['min']:.2f} to {info['max']:.2f}")
                print(f"   Mean: {info['mean']:.2f} (±{info['std']:.2f})")
                print(f"   Unique values: {info['unique_values']}")
            else:
                print(f"   Categories: {info['categories']}")
                print(f"   Distribution: {info['value_counts']}")
        
        return feature_info

# Convenience function to load data quickly
def load_drug_data(data_path="data/raw/drug200.csv"):
    """
    Quick function to load the drug dataset.
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    loader = DataLoader(data_path)
    return loader.load_data()

def load_processed_data():
    """
    Load the preprocessed training, validation, and test data.
    
    This function loads the data that has already been preprocessed and split
    into training, validation, and test sets.
    
    Returns:
        tuple: (train_data, val_data, test_data) as pandas DataFrames
        
    Raises:
        FileNotFoundError: If processed data files don't exist
        Exception: If there's an error loading the processed data
    """
    try:
        print("📂 Loading preprocessed data...")
        
        # Define file paths
        train_path = "data/processed/train.csv"
        val_path = "data/processed/validation.csv"
        test_path = "data/processed/test.csv"
        
        # Check if all files exist
        missing_files = []
        for path in [train_path, val_path, test_path]:
            if not Path(path).exists():
                missing_files.append(path)
        
        if missing_files:
            raise FileNotFoundError(
                f"Processed data files not found: {missing_files}\n"
                "Please run the preprocessing pipeline first."
            )
        
        # Load the data
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        test_data = pd.read_csv(test_path)
        
        print(f"✅ Preprocessed data loaded successfully!")
        print(f"   Training data: {train_data.shape}")
        print(f"   Validation data: {val_data.shape}")
        print(f"   Test data: {test_data.shape}")
        
        # Check if target column exists and rename it if needed
        target_columns = ['Drug', 'Drug_encoded', 'target']
        target_col = None
        
        for col in target_columns:
            if col in train_data.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError(f"No target column found. Expected one of: {target_columns}")
        
        # Rename target column to 'Drug' for consistency
        if target_col != 'Drug':
            print(f"📝 Renaming target column '{target_col}' to 'Drug'")
            train_data = train_data.rename(columns={target_col: 'Drug'})
            val_data = val_data.rename(columns={target_col: 'Drug'})
            test_data = test_data.rename(columns={target_col: 'Drug'})
        
        # Validate expected feature columns exist
        expected_feature_cols = ['num__Age', 'num__Na_to_K', 'cat__Sex_M', 
                               'cat__BP_LOW', 'cat__BP_NORMAL', 'cat__Cholesterol_NORMAL']
        
        for name, data in [("training", train_data), ("validation", val_data), ("test", test_data)]:
            missing_cols = set(expected_feature_cols) - set(data.columns)
            if missing_cols:
                print(f"⚠️  Warning: {name} data missing feature columns: {missing_cols}")
        
        print(f"✅ Data validation completed!")
        print(f"   Target column: 'Drug' (encoded values: {sorted(train_data['Drug'].unique())})")
        print(f"   Feature columns: {len(expected_feature_cols)} features")
        
        return train_data, val_data, test_data
        
    except Exception as e:
        print(f"❌ Error loading processed data: {str(e)}")
        print("💡 Tip: Make sure you've run the preprocessing step first!")
        raise