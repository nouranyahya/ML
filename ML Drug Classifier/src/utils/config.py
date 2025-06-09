"""
Configuration utilities for the drug classification project.

This module handles loading and managing configuration settings from config.yaml.
Think of this as the "settings manager" for our entire project.
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the drug classification project."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        
        # Data paths
        self.DATA_RAW_PATH = "data/raw/drug200.csv"
        self.DATA_PROCESSED_PATH = "data/processed/"
        
        # Model paths
        self.MODELS_PATH = "models/saved_models/"
        self.HYPERPARAMS_PATH = "models/hyperparameters/"
        
        # Results paths
        self.RESULTS_PATH = "results/"
        self.PLOTS_PATH = "results/plots/"
        self.METRICS_PATH = "results/metrics/"
        self.REPORTS_PATH = "results/reports/"
        
        # Data preprocessing
        self.TEST_SIZE = 0.2
        self.VALIDATION_SIZE = 0.1
        self.RANDOM_STATE = 42
        
        # Model parameters
        self.MODEL_PARAMS = {
            'logistic_regression': {
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': self.RANDOM_STATE
            },
            'knn': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto',
                'metric': 'minkowski',
                'p': 2
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': self.RANDOM_STATE
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': self.RANDOM_STATE,
                'n_jobs': -1
            },
            'neural_network': {
                'hidden_layers': [64, 32, 16],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32
            }
        }
        
        # Visualization settings
        self.PLOT_STYLE = 'seaborn-v0_8'
        self.FIGURE_SIZE = (10, 6)
        self.DPI = 300
        
        # Logging
        self.LOG_LEVEL = 'INFO'
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model.
        
        Parameters:
        - model_name: Name of the model
        
        Returns:
        - Dictionary of model parameters
        """
        return self.MODEL_PARAMS.get(model_name, {})
    
    def update_param(self, param_name: str, value: Any):
        """
        Update a configuration parameter.
        
        Parameters:
        - param_name: Name of the parameter
        - value: New value for the parameter
        """
        setattr(self, param_name, value)
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.DATA_PROCESSED_PATH,
            self.MODELS_PATH,
            self.HYPERPARAMS_PATH,
            self.RESULTS_PATH,
            self.PLOTS_PATH,
            self.METRICS_PATH,
            self.REPORTS_PATH,
            f"{self.PLOTS_PATH}/data_exploration",
            f"{self.PLOTS_PATH}/model_performance",
            f"{self.PLOTS_PATH}/feature_importance"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get(self, key, default=None):
        """
        Get a configuration value using dot notation.
        
        Parameters:
        - key: Configuration key (supports dot notation like 'data.raw_data')
        - default: Default value if key is not found
        
        Returns:
        - Configuration value or default
        """
        # Handle dot notation (e.g., 'data.raw_data')
        if '.' in key:
            # Map common dot notation patterns to actual attributes
            key_mappings = {
                'data.raw_data': 'DATA_RAW_PATH',
                'data.processed': 'DATA_PROCESSED_PATH',
                'models.path': 'MODELS_PATH',
                'results.path': 'RESULTS_PATH',
                'random_state': 'RANDOM_STATE',
                'test_size': 'TEST_SIZE'
            }
            
            if key in key_mappings:
                attr_name = key_mappings[key]
                return getattr(self, attr_name, default)
            else:
                return default
        else:
            # Direct attribute access
            return getattr(self, key.upper(), default)
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Parameters:
        - key: Configuration key
        - value: Value to set
        """
        if '.' in key:
            # Handle dot notation
            key_mappings = {
                'data.raw_data': 'DATA_RAW_PATH',
                'data.processed': 'DATA_PROCESSED_PATH',
                'models.path': 'MODELS_PATH',
                'results.path': 'RESULTS_PATH',
                'random_state': 'RANDOM_STATE',
                'test_size': 'TEST_SIZE'
            }
            
            if key in key_mappings:
                attr_name = key_mappings[key]
                setattr(self, attr_name, value)
        else:
            setattr(self, key.upper(), value)
    
    def __str__(self):
        """String representation of the configuration."""
        return f"Config(random_state={self.RANDOM_STATE}, test_size={self.TEST_SIZE})"

# Global configuration instance
config = Config()