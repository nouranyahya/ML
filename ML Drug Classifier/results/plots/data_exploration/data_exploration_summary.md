# Data Exploration Summary Report

## Dataset Overview
- **Total Samples:** 200
- **Total Features:** 5
- **Target Variable:** Drug (5 classes)
- **Missing Values:** 0
- **Duplicate Rows:** 0

## Feature Summary
### Numerical Features
- **Age:** Range [15, 74], Mean: 44.3, Std: 16.5
- **Na_to_K:** Range [6.27, 38.25], Mean: 16.08, Std: 7.22

### Categorical Features
- **Sex:** {'M': 104, 'F': 96}
- **Blood Pressure:** {'HIGH': 77, 'LOW': 64, 'NORMAL': 59}
- **Cholesterol:** {'HIGH': 103, 'NORMAL': 97}

## Target Variable Analysis
### Drug Distribution
Drug
DrugY    91
drugX    54
drugA    23
drugC    16
drugB    16

**Class Balance Assessment:**
- Imbalance Ratio: 5.69
- Status: Highly Imbalanced

## Data Quality Assessment
### Outliers Detected
- Age: 0 outliers
- Na_to_K: 8 outliers

### Data Quality Score
- Missing Values: ✅ None
- Duplicates: ✅ None
- Outliers: ✅ Minimal
- Class Balance: ❌ Highly Imbalanced

## Key Insights
1. **Dataset Size:** The dataset contains 200 samples, which is sufficient for machine learning but not very large.
2. **Feature Quality:** All features are complete with no missing values.
3. **Target Distribution:** Classes are imbalanced with ratio 5.69.
4. **Feature Relationships:** Age and Na_to_K ratio appear to be important predictors based on their distributions across drug classes.

## Recommendations for Modeling
1. **Data Preprocessing:** Standard scaling recommended for numerical features.
2. **Class Imbalance:** Consider techniques like SMOTE or class weights.
3. **Feature Engineering:** Consider creating interaction features between categorical variables.
4. **Model Selection:** The dataset size and feature types suggest tree-based models and neural networks may work well.

## Generated Visualizations
1. `01_overview_dashboard.png` - Comprehensive dataset overview
2. `02_feature_distributions.png` - Individual feature distributions
3. `03_target_analysis.png` - Target variable analysis
4. `04_correlation_analysis.png` - Feature correlation analysis
5. `05_data_quality_report.png` - Data quality assessment
6. `06_interactive_dashboard.html` - Interactive exploration dashboard
