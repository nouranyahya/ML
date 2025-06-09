"""
Data preprocessing for the drug classification project.

This module handles all data preprocessing steps including:
- Encoding categorical variables (converting text to numbers)
- Scaling numerical features (making all numbers on similar scales)
- Splitting data into train/test sets

Think of this as the "data preparation kitchen" where we transform raw ingredients
into a format that our machine learning models can digest.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.helpers import ensure_dir, print_dataset_info

class DrugDataPreprocessor:
    """
    A class to handle all preprocessing steps for the drug classification dataset.
    
    This class takes raw data and transforms it into a format suitable for machine learning.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.preprocessor = None
        self.feature_names = None
        self.target_encoder = LabelEncoder()
        
    def prepare_features_and_target(self, df):
        """
        Separate features (input variables) and target (what we want to predict).
        
        Args:
            df (pandas.DataFrame): The raw dataset
            
        Returns:
            tuple: (X, y) where X is features and y is target
            
        In our case:
        - Features (X): Age, Sex, BP, Cholesterol, Na_to_K
        - Target (y): Drug (what we want to predict)
        """
        # Features are all columns except 'Drug'
        X = df.drop('Drug', axis=1)
        
        # Target is the 'Drug' column
        y = df['Drug']
        
        print("📊 Features and Target separated:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Feature columns: {list(X.columns)}")
        print(f"   Target classes: {sorted(y.unique())}")
        
        return X, y
    
    def create_preprocessor(self, X):
        """
        Create preprocessing pipeline for features.
        
        Args:
            X (pandas.DataFrame): Feature dataset
            
        This function sets up different preprocessing for different types of data:
        - Numerical data (Age, Na_to_K): Scale to standard range
        - Categorical data (Sex, BP, Cholesterol): Convert to numbers
        """
        # Identify column types
        numerical_columns = ['Age', 'Na_to_K']
        categorical_columns = ['Sex', 'BP', 'Cholesterol']
        
        print("🔧 Setting up preprocessing pipeline:")
        print(f"   Numerical columns: {numerical_columns}")
        print(f"   Categorical columns: {categorical_columns}")
        
        # Create preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                # For numerical columns: scale to standard range (mean=0, std=1)
                ('num', StandardScaler(), numerical_columns),
                # For categorical columns: convert to one-hot encoding
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
            ],
            remainder='passthrough'  # Keep any other columns as-is
        )
        
        return self.preprocessor
    
    def fit_transform_features(self, X_train, X_test=None):
        """
        Fit the preprocessor on training data and transform both train and test data.
        
        Args:
            X_train (pandas.DataFrame): Training features
            X_test (pandas.DataFrame): Test features (optional)
            
        Returns:
            tuple: (X_train_processed, X_test_processed) or just X_train_processed
            
        Why fit on training data only?
        We learn the preprocessing parameters (like mean and std) from training data only,
        then apply these same parameters to test data. This prevents "data leakage"
        where test data accidentally influences our model.
        """
        print("🔄 Preprocessing features...")
        
        # Create preprocessor if not already created
        if self.preprocessor is None:
            self.create_preprocessor(X_train)
        
        # Fit and transform training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Get feature names after preprocessing
        self._get_feature_names()
        
        print(f"   Training features shape: {X_train.shape} → {X_train_processed.shape}")
        
        if X_test is not None:
            # Transform test data using the same preprocessor
            X_test_processed = self.preprocessor.transform(X_test)
            print(f"   Test features shape: {X_test.shape} → {X_test_processed.shape}")
            return X_train_processed, X_test_processed
        
        return X_train_processed
    
    def _get_feature_names(self):
        """
        Get the names of features after preprocessing.
        
        After preprocessing, we might have more columns due to one-hot encoding.
        This function helps us keep track of what each column represents.
        """
        feature_names = []
        
        # Get feature names from the preprocessor
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            # Fallback for older scikit-learn versions
            numerical_features = ['Age', 'Na_to_K']
            
            # Categorical features after one-hot encoding
            categorical_features = []
            for col in ['Sex', 'BP', 'Cholesterol']:
                if col == 'Sex':
                    categorical_features.append('Sex_M')  # Only M, F is dropped
                elif col == 'BP':
                    categorical_features.extend(['BP_LOW', 'BP_NORMAL'])  # HIGH is dropped
                elif col == 'Cholesterol':
                    categorical_features.append('Cholesterol_NORMAL')  # HIGH is dropped
            
            feature_names = numerical_features + categorical_features
        
        self.feature_names = feature_names
        print(f"   Processed feature names: {feature_names}")
    
    def encode_target(self, y_train, y_test=None):
        """
        Encode target variable (convert drug names to numbers).
        
        Args:
            y_train: Training target values
            y_test: Test target values (optional)
            
        Returns:
            Encoded target values
            
        Why encode the target?
        Many machine learning algorithms work better with numerical targets.
        For example: DrugA=0, DrugB=1, drugC=2, drugX=3, DrugY=4
        """
        print("🏷️  Encoding target variable...")
        
        # Fit and transform training target
        y_train_encoded = self.target_encoder.fit_transform(y_train)
        
        # Show the mapping
        classes = self.target_encoder.classes_
        print(f"   Target encoding mapping:")
        for i, drug in enumerate(classes):
            print(f"     {drug} → {i}")
        
        if y_test is not None:
            # Transform test target using the same encoder
            y_test_encoded = self.target_encoder.transform(y_test)
            return y_train_encoded, y_test_encoded
        
        return y_train_encoded
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.1, random_state=42):
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Features
            y: Target variable
            test_size (float): Proportion of data for testing (0.2 = 20%)
            validation_size (float): Proportion of data for validation (0.1 = 10%)
            random_state (int): Random seed for reproducible results
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            
        Why three splits?
        - Training (70%): Teach the model
        - Validation (10%): Tune model parameters
        - Test (20%): Final evaluation (never seen during training)
        """
        print("✂️  Splitting data into train/validation/test sets...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate validation from remaining data
        # Adjust validation size relative to remaining data
        val_size_adjusted = validation_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"   Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_complete_pipeline(self, df, save_processed=True, output_dir="data/processed/"):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df (pandas.DataFrame): Raw dataset
            save_processed (bool): Whether to save processed data
            output_dir (str): Directory to save processed data
            
        Returns:
            dict: All processed datasets and encoders
            
        This is the main function that coordinates all preprocessing steps.
        """
        print("\n" + "="*60)
        print("🚀 STARTING COMPLETE PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Prepare features and target
        X, y = self.prepare_features_and_target(df)
        
        # Step 2: Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 3: Preprocess features
        X_train_processed, X_test_processed = self.fit_transform_features(X_train, X_test)
        X_val_processed = self.preprocessor.transform(X_val)
        
        # Step 4: Encode target variables
        y_train_encoded, y_test_encoded = self.encode_target(y_train, y_test)
        y_val_encoded = self.target_encoder.transform(y_val)
        
        # Create processed DataFrames
        feature_names = self.feature_names if self.feature_names is not None else [f"feature_{i}" for i in range(X_train_processed.shape[1])]        
        train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        train_df['Drug_encoded'] = y_train_encoded
        
        val_df = pd.DataFrame(X_val_processed, columns=feature_names)
        val_df['Drug_encoded'] = y_val_encoded
        
        test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        test_df['Drug_encoded'] = y_test_encoded
        
        # Save processed data if requested
        if save_processed:
            self._save_processed_data(train_df, val_df, test_df, output_dir)
        
        print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        
        # Return everything in a organized dictionary
        return {
            'X_train': X_train_processed,
            'X_val': X_val_processed,
            'X_test': X_test_processed,
            'y_train': y_train_encoded,
            'y_val': y_val_encoded,
            'y_test': y_test_encoded,
            'y_train_original': y_train,
            'y_val_original': y_val,
            'y_test_original': y_test,
            'feature_names': feature_names,
            'target_classes': self.target_encoder.classes_,
            'preprocessor': self.preprocessor,
            'target_encoder': self.target_encoder,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        }
    
    def _save_processed_data(self, train_df, val_df, test_df, output_dir):
        """
        Save processed datasets to CSV files.
        
        Args:
            train_df, val_df, test_df: Processed DataFrames
            output_dir (str): Directory to save files
        """
        ensure_dir(output_dir)
        
        train_path = os.path.join(output_dir, "train.csv")
        val_path = os.path.join(output_dir, "validation.csv")
        test_path = os.path.join(output_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n💾 Processed data saved:")
        print(f"   Training data: {train_path}")
        print(f"   Validation data: {val_path}")
        print(f"   Test data: {test_path}")

# Convenience function for quick preprocessing
def preprocess_drug_data(df):
    """
    Quick function to preprocess the drug dataset.
    
    Args:
        df (pandas.DataFrame): Raw dataset
        
    Returns:
        dict: Processed data and encoders
    """
    preprocessor = DrugDataPreprocessor()
    return preprocessor.preprocess_complete_pipeline(df)