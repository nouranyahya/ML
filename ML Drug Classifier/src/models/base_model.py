"""
Base model class for the drug classification project.

This module provides a common interface that all our machine learning models will follow.
Think of this as a "blueprint" or "template" that ensures all our models work consistently.

Why use a base class?
- Consistency: All models have the same methods (train, predict, evaluate)
- Code reuse: Common functionality is written once
- Easy comparison: All models can be used interchangeably
"""
from abc import ABC, abstractmethod
import os
from datetime import datetime

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    Defines the interface that all models must implement.
    """
    
    def __init__(self):
        self.model = None
        self.model_name = "Base Model"
        self.training_info = {}
    
    @abstractmethod
    def create_model(self):
        """Create and return the model instance."""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_val: Validation features (optional)
        - y_val: Validation labels (optional)
        
        Returns:
        - Dictionary with training results
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        - X: Features to predict on
        
        Returns:
        - Predictions
        """
        pass
    
    @abstractmethod
    def save(self, filepath):
        """
        Save the trained model.
        
        Parameters:
        - filepath: Directory path to save the model
        
        Returns:
        - Full path to the saved model
        """
        pass
    
    @abstractmethod
    def load(self, filepath):
        """
        Load a trained model.
        
        Parameters:
        - filepath: Path to the model file
        
        Returns:
        - Loaded model
        """
        pass
    
    def get_model_info(self):
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'is_trained': self.model is not None,
            'training_info': self.training_info
        }
    
    def _ensure_directory_exists(self, directory):
        """Ensure a directory exists, create if it doesn't."""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def _get_timestamp(self):
        """Get current timestamp as string."""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")