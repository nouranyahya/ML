"""
Logistic Regression model for drug classification.

WHAT IS LOGISTIC REGRESSION?
Logistic Regression is like drawing lines (or curves) to separate different groups.
Imagine you're sorting M&Ms by color - you'd draw boundaries between red, blue, green areas.

HOW IT WORKS:
1. It looks at the relationship between features (age, blood pressure, etc.) and the outcome (which drug)
2. It finds the best "decision boundaries" that separate the classes
3. For new patients, it calculates probabilities for each drug class

WHEN TO USE IT:
- Good as a baseline model (simple and fast)
- Works well when classes are linearly separable
- Provides probability estimates (confidence in predictions)
- Easy to interpret (you can see which features matter most)
"""
import joblib
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier implementation."""
    
    def __init__(self, C=1.0, solver='lbfgs', max_iter=1000, random_state=42):
        """
        Initialize Logistic Regression model.
        
        Parameters:
        - C: Inverse of regularization strength
        - solver: Algorithm to use in optimization
        - max_iter: Maximum number of iterations
        - random_state: Random state for reproducibility
        """
        super().__init__()
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.model_name = "Logistic Regression"
        
    def create_model(self):
        """Create and return a Logistic Regression model."""
        print(f"🔧 Creating Logistic Regression with parameters:")
        print(f"   C: {self.C}")
        print(f"   solver: {self.solver}")
        print(f"   max_iter: {self.max_iter}")
        print(f"   random_state: {self.random_state}")
        
        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Logistic Regression model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_val: Validation features (optional)
        - y_val: Validation labels (optional)
        """
        if self.model is None:
            self.create_model()
            
        print(f"🚀 Training Logistic Regression...")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Number of classes: {len(set(y_train))}")
        
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
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'training_samples': X_train.shape[0],
            'features': X_train.shape[1],
            'classes': len(set(y_train))
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
        model_filename = f"logistic_regression_{timestamp}.pkl"
        full_path = f"{filepath}/{model_filename}"
        
        # Save the model
        joblib.dump(self.model, full_path)
        print(f"📊 Logistic Regression model saved to: {full_path}")
        
        return full_path
    
    def load(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)
        return self.model
    
    def get_params(self):
        """Get model parameters."""
        return {
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }
    
    def get_coefficients(self):
        """Get model coefficients if available."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return {
            'coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist()
        }