"""
Model evaluation utilities.

This package contains modules for:
- Computing evaluation metrics
- Comparing multiple models
- Statistical significance testing
"""

from .metrics import ModelEvaluator
from .model_comparison import ModelComparison

__all__ = ['ModelEvaluator', 'ModelComparison']
