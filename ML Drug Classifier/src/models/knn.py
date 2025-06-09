"""
K-Nearest Neighbors (KNN) model for drug classification.

WHAT IS KNN?
KNN is like asking your neighbors for advice! It looks at the K closest examples
in the training data and predicts based on what the majority of those neighbors are.

HOW IT WORKS:
1. When predicting for a new patient, it finds the K most similar patients from training data
2. It looks at what drugs those similar patients were prescribed
3. It predicts the most common drug among those K neighbors

EXAMPLE:
If K=5 and the 5 most similar patients took: DrugA, DrugA, DrugB, DrugA, DrugC
Then it predicts DrugA (appears 3 out of 5 times)

PROS:
- Simple and intuitive
- No assumptions about data distribution
- Works well with irregular decision boundaries
- Good for datasets where similar inputs should have similar outputs

CONS:
- Can be slow with large datasets (has to compare with all training data)
- Sensitive to irrelevant features
- Sensitive to the scale of features (why we need to normalize data)
"""

import joblib
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from .base_model import BaseModel

class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier implementation."""
    
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', 
                 metric='minkowski', p=2):
        """
        Initialize KNN model.
        
        Parameters:
        - n_neighbors: Number of neighbors to use
        - weights: Weight function used in prediction
        - algorithm: Algorithm used to compute nearest neighbors
        - metric: Distance metric to use
        - p: Power parameter for Minkowski metric
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.model = None
        self.model_name = "K-Nearest Neighbors"
        
    def create_model(self):
        """Create and return a KNN model."""
        print(f"🔧 Creating K-Nearest Neighbors with parameters:")
        print(f"   n_neighbors: {self.n_neighbors}")
        print(f"   weights: {self.weights}")
        print(f"   algorithm: {self.algorithm}")
        print(f"   metric: {self.metric}")
        print(f"   p: {self.p}")
        
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p
        )
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the KNN model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_val: Validation features (optional)
        - y_val: Validation labels (optional)
        """
        if self.model is None:
            self.create_model()
            
        print(f"🚀 Training K-Nearest Neighbors...")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Number of classes: {len(set(y_train))}")
        print(f"   Number of neighbors (K): {self.n_neighbors}")
        
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
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'p': self.p,
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
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"k-nearest_neighbors_{timestamp}.pkl"
        full_path = f"{filepath}/{model_filename}"
        
        # Save the model
        joblib.dump(self.model, full_path)
        print(f"🤖 K-Nearest Neighbors model saved to: {full_path}")
        
        return full_path
    
    def load(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)
        return self.model
    
    def get_params(self):
        """Get model parameters."""
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'p': self.p
        }