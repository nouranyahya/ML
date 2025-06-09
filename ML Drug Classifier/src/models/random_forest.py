"""
Random Forest model for drug classification.

WHAT IS RANDOM FOREST?
Random Forest is like asking multiple doctors for their opinion and taking a vote!
It creates many decision trees (each like a doctor with their own decision process)
and combines their predictions to make a final decision.

HOW IT WORKS:
1. Creates many decision trees (typically 100-1000)
2. Each tree is trained on a random subset of the data
3. Each tree uses only a random subset of features at each decision point
4. For prediction, all trees vote and majority wins
5. The randomness prevents overfitting and makes the model robust

DECISION TREE EXAMPLE:
Tree might ask: "Is age > 50?" → Yes: "Is BP = HIGH?" → Yes: "Predict DrugY"
                                  → No: "Predict DrugX"
               → No: "Is Cholesterol = HIGH?" → Yes: "Predict DrugA"
                                              → No: "Predict DrugB"

WHY "RANDOM"?
- Random sampling of data (bootstrap sampling)
- Random selection of features at each split
- This randomness makes the forest more robust and less prone to overfitting

PROS:
- Very robust and accurate
- Handles both numerical and categorical features
- Provides feature importance scores
- Less prone to overfitting than single decision trees
- Can handle missing values
- Fast training and prediction

CONS:
- Can be memory intensive with many trees
- Less interpretable than single decision tree
- Can still overfit with very noisy data
"""

import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest classifier implementation."""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1):
        """
        Initialize Random Forest model.
        
        Parameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum samples required to split a node
        - min_samples_leaf: Minimum samples required at a leaf node
        - max_features: Number of features to consider when looking for the best split
        - random_state: Random state for reproducibility
        - n_jobs: Number of jobs to run in parallel
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.model_name = "Random Forest"
        
    def create_model(self):
        """Create and return a Random Forest model."""
        print(f"🔧 Creating Random Forest with parameters:")
        print(f"   n_estimators: {self.n_estimators}")
        print(f"   max_depth: {self.max_depth}")
        print(f"   min_samples_split: {self.min_samples_split}")
        print(f"   min_samples_leaf: {self.min_samples_leaf}")
        print(f"   max_features: {self.max_features}")
        print(f"   random_state: {self.random_state}")
        print(f"   n_jobs: {self.n_jobs}")
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Random Forest model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_val: Validation features (optional)
        - y_val: Validation labels (optional)
        """
        if self.model is None:
            self.create_model()
            
        print(f"🚀 Training Random Forest...")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Number of classes: {len(set(y_train))}")
        print(f"   Number of trees: {self.n_estimators}")
        print(f"   Max features per split: {self.max_features}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Calculate validation accuracy if validation data is provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
        
        print(f"✅ Training completed!")
        print(f"   Training accuracy: {train_accuracy:.4f}")
        if val_accuracy is not None:
            print(f"   Validation accuracy: {val_accuracy:.4f}")
        
        # Store training information
        self.training_info = {
            'train_accuracy': float(train_accuracy),
            'val_accuracy': float(val_accuracy) if val_accuracy is not None else None,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'training_samples': X_train.shape[0],
            'features': X_train.shape[1],
            'classes': len(set(y_train)),
            'feature_importances': [float(x) for x in self.model.feature_importances_]
        }
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'model': self.model
        }
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Ensure directory exists
        os.makedirs(filepath, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"random_forest_{timestamp}.pkl"
        full_path = f"{filepath}/{model_filename}"
        
        # Save the model
        joblib.dump(self.model, full_path)
        print(f"🌳 Random Forest model saved to: {full_path}")
        
        return full_path
    
    def load(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)
        return self.model
    
    def get_params(self):
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return {
            'feature_importances': self.model.feature_importances_.tolist(),
            'sorted_indices': self.model.feature_importances_.argsort()[::-1].tolist()
        }