"""
Support Vector Machine (SVM) model for drug classification.

WHAT IS SVM?
SVM is like finding the best way to separate different groups with a line (or curve).
Imagine you have red and blue marbles mixed together on a table - SVM finds the
best line that separates them with the maximum gap (margin) between groups.

HOW IT WORKS:
1. It tries to find the optimal boundary (hyperplane) that separates classes
2. It maximizes the "margin" - the distance between the boundary and the closest points
3. The closest points to the boundary are called "support vectors" (hence the name)
4. Uses "kernel tricks" to handle complex, non-linear boundaries

KERNEL TYPES:
- Linear: Straight line boundaries (fast, good for linearly separable data)
- RBF (Radial Basis Function): Curved boundaries (most common, handles complex patterns)
- Polynomial: Polynomial-shaped boundaries
- Sigmoid: S-shaped boundaries

PROS:
- Very effective for high-dimensional data
- Memory efficient (only stores support vectors)
- Versatile with different kernel functions
- Works well with small to medium datasets

CONS:
- Can be slow on large datasets
- Sensitive to feature scaling
- No probability estimates by default (need probability=True)
- Hard to interpret (black box)
"""

import joblib
import os
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from .base_model import BaseModel

class SVMModel(BaseModel):
    """Support Vector Machine classifier implementation."""
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42):
        """
        Initialize SVM model.
        
        Parameters:
        - C: Regularization parameter
        - kernel: Kernel type to be used
        - gamma: Kernel coefficient
        - probability: Whether to enable probability estimates
        - random_state: Random state for reproducibility
        """
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.model = None
        self.model_name = "Support Vector Machine"
        
    def create_model(self):
        """Create and return an SVM model."""
        print(f"🔧 Creating Support Vector Machine with parameters:")
        print(f"   C: {self.C}")
        print(f"   kernel: {self.kernel}")
        print(f"   gamma: {self.gamma}")
        print(f"   probability: {self.probability}")
        print(f"   random_state: {self.random_state}")
        
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state
        )
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the SVM model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_val: Validation features (optional)
        - y_val: Validation labels (optional)
        """
        if self.model is None:
            self.create_model()
            
        print(f"🚀 Training Support Vector Machine...")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Number of classes: {len(set(y_train))}")
        print(f"   Kernel: {self.kernel}")
        print(f"   C parameter: {self.C}")
        
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
        
        # Get support vector information
        n_support_vectors = len(self.model.support_)
        support_vectors_per_class = self.model.n_support_
        
        print(f"   Total support vectors: {n_support_vectors}")
        print(f"   Support vectors per class: {support_vectors_per_class}")
        
        # Store training information
        self.training_info = {
            'train_accuracy': float(train_accuracy),
            'val_accuracy': float(val_accuracy) if val_accuracy is not None else None,
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'probability': self.probability,
            'random_state': self.random_state,
            'training_samples': X_train.shape[0],
            'features': X_train.shape[1],
            'classes': len(set(y_train)),
            'n_support_vectors': int(n_support_vectors),
            'support_vectors_per_class': [int(x) for x in support_vectors_per_class]
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
        if not self.probability:
            raise ValueError("Probability estimation is not enabled. Set probability=True when creating the model.")
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Ensure directory exists
        os.makedirs(filepath, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"support_vector_machine_{timestamp}.pkl"
        full_path = f"{filepath}/{model_filename}"
        
        # Save the model
        joblib.dump(self.model, full_path)
        print(f"🤖 Support Vector Machine model saved to: {full_path}")
        
        return full_path
    
    def load(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)
        return self.model
    
    def get_params(self):
        """Get model parameters."""
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'probability': self.probability,
            'random_state': self.random_state
        }
    
    def get_support_vectors_info(self):
        """Get information about support vectors."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return {
            'n_support_vectors': len(self.model.support_),
            'support_vectors_per_class': self.model.n_support_.tolist(),
            'support_vector_indices': self.model.support_.tolist()
        }