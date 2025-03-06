# Credit Card Fraud Detection

## Introduction
This project implements machine learning models to identify fraudulent credit card transactions. Using Decision Trees and Support Vector Machines (SVM), the system analyzes transaction patterns to detect potential fraud. The project leverages a real dataset from Kaggle containing credit card transactions, where only 0.172% of transactions are fraudulent.

## Project Overview
- **Classification Problem**: Binary classification to determine if a transaction is fraudulent (1) or legitimate (0)
- **Data Imbalance**: Only 492 out of 284,807 transactions are fraudulent (0.172%)
- **Models Implemented**: Decision Tree and Support Vector Machine
- **Performance Metric**: ROC-AUC score

## Technologies Used
- Python
- pandas
- scikit-learn
- matplotlib

## Dataset
The dataset contains credit card transactions made by European cardholders in September 2013. Due to confidentiality, the original features have been transformed using PCA. The only features not transformed are 'Time' and 'Amount'.

Dataset source: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Implementation
The implementation includes:
1. Data preprocessing (standardization and normalization)
2. Training Decision Tree and SVM models
3. Handling class imbalance through sample weighting
4. Model evaluation using ROC-AUC scores
5. Feature importance analysis

## Key Visualizations
- Class distribution pie chart
- Feature correlation analysis
- ROC curves for model performance comparison
- Top correlated features for fraud detection

## Results
The models are evaluated using ROC-AUC scores, which measure their ability to distinguish between fraudulent and legitimate transactions across different probability thresholds.

## How to Run
1. Ensure you have the required dependencies installed
2. Download the dataset from Kaggle
3. Run the `ccfdetect.py` script

## Future Work
- Implement additional models (Random Forests, Gradient Boosting)
- Explore feature engineering techniques
- Implement cost-sensitive learning approaches
- Deploy as a real-time fraud detection system