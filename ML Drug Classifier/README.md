"""
Drug Classification Project README

A comprehensive machine learning project for classifying drug types based on patient characteristics.
"""

# Drug Classification with Machine Learning

A comprehensive machine learning project that predicts the appropriate drug for patients based on their medical characteristics using multiple algorithms and evaluation techniques.

## 🎯 Project Overview

This project implements and compares 5 different machine learning algorithms to classify drugs based on patient features:
- Age
- Sex (M/F)
- Blood Pressure (HIGH/NORMAL/LOW)
- Cholesterol (HIGH/NORMAL)
- Na_to_K ratio (Sodium to Potassium ratio)

**Target Classes:** DrugA, DrugB, drugC, drugX, DrugY

## 🏗️ Project Structure

```
drug-classification-project/
├── data/
│   ├── raw/
│   │   └── drug200.csv
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       └── validation.csv
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   └── neural_network.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── model_comparison.py
│   ├── visualization/
│   │   ├── data_plots.py
│   │   └── results_plots.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── scripts/
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── generate_plots.py
│   └── run_pipeline.py
├── models/
│   └── saved_models/
├── results/
│   ├── plots/
│   ├── metrics/
│   └── reports/
├── requirements.txt
├── config.yaml
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. **Clone/Download the project**
   ```bash
   git clone <repository-url>
   cd drug-classification-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your dataset**
   Place `drug200.csv` in the `data/raw/` directory

### Running the Complete Pipeline

**Option 1: Run everything at once**
```bash
python scripts/run_pipeline.py
```

**Option 2: Run steps individually**
```bash
# Step 1: Train all models
python scripts/train_models.py

# Step 2: Evaluate models
python scripts/evaluate_models.py

# Step 3: Generate visualizations
python scripts/generate_plots.py
```

## 🤖 Models Implemented

1. **Logistic Regression** - Linear classification with probability estimates
2. **K-Nearest Neighbors (KNN)** - Instance-based learning
3. **Support Vector Machine (SVM)** - Maximum margin classification
4. **Random Forest** - Ensemble of decision trees
5. **Neural Network** - Deep learning with TensorFlow

## 📊 Features & Capabilities

### Data Processing
- Automated data loading and validation
- Feature encoding (categorical → numerical)
- Data scaling and normalization
- Train/validation/test splitting

### Model Training
- Hyperparameter tuning for all models
- Cross-validation for robust evaluation
- Model persistence (save/load)
- Training history tracking

### Evaluation & Comparison
- Comprehensive metrics (accuracy, precision, recall, F1)
- Statistical significance testing
- ROC curves and confusion matrices
- Feature importance analysis
- Model ranking and recommendations

### Visualizations
- Interactive data exploration dashboards
- Model performance comparisons
- Feature importance plots
- Prediction analysis
- Learning curves

## 📈 Results & Outputs

After running the pipeline, you'll find:

### Reports (`results/reports/`)
- `final_project_summary.md` - Complete project overview
- `model_comparison_report.md` - Detailed model comparison
- `data_exploration_summary.md` - Data analysis insights

### Visualizations (`results/plots/`)
- Data exploration dashboards
- Model performance charts
- Interactive HTML dashboards
- ROC curves and confusion matrices

### Models (`models/saved_models/`)
- Trained model files for all algorithms
- Model metadata and parameters

### Metrics (`results/evaluation/`)
- Detailed evaluation results for each model
- Cross-validation scores
- Statistical test results

## 🎓 Learning Objectives

This project teaches:

### Machine Learning Concepts
- **Supervised Learning**: Classification with labeled data
- **Model Comparison**: Systematic evaluation of algorithms
- **Hyperparameter Tuning**: Optimizing model performance
- **Cross-Validation**: Robust model assessment
- **Feature Engineering**: Data preprocessing techniques

### Practical Skills
- **Python Programming**: Object-oriented design patterns
- **Data Science Workflow**: End-to-end ML pipeline
- **Visualization**: Creating insightful charts and dashboards
- **Model Deployment**: Saving and loading trained models
- **Documentation**: Code organization and reporting

### Industry Best Practices
- **Code Organization**: Modular, reusable components
- **Version Control**: Git-friendly project structure
- **Reproducibility**: Configurable parameters and random seeds
- **Testing**: Validation and error handling
- **Reporting**: Professional documentation and visualizations

## 🔧 Configuration

Edit `config.yaml` to customize:
- File paths
- Model parameters
- Training settings
- Output preferences

## 📋 Requirements

See `requirements.txt` for full dependencies. Key packages:
- pandas, numpy - Data manipulation
- scikit-learn - ML algorithms
- tensorflow - Neural networks
- matplotlib, seaborn, plotly - Visualizations
- jupyter - Interactive development

## 🎯 Model Performance

The project provides comprehensive comparison of all models including:
- Accuracy and F1-scores
- Training and prediction speed
- Statistical significance tests
- Recommendations for different use cases

## 🚀 Next Steps

After completing this project, consider:
1. **Feature Engineering**: Create new features from existing ones
2. **Ensemble Methods**: Combine multiple models
3. **Deep Learning**: Experiment with more complex neural networks
4. **Deployment**: Create a web API for real-time predictions
5. **Clinical Validation**: Test with real medical data (with proper permissions)

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://tensorflow.org/)
- [Machine Learning Best Practices](https://ml-ops.org/)

## ⚠️ Important Notes

- This is an educational project - not for medical diagnosis
- Always validate ML models with domain experts
- Ensure data privacy and compliance in real applications
- Consider ethical implications of automated medical decisions

---

**Happy Learning! 🎉**

This project provides a solid foundation in machine learning while teaching industry best practices and comprehensive evaluation techniques.