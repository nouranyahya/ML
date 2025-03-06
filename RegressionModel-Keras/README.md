# Concrete Strength Prediction Model

## Overview
This project implements a neural network regression model to predict concrete strength based on various input features. The implementation explores different model configurations to optimize performance, including feature normalization, varying the number of training epochs, and adjusting network depth.

## Dataset
The model uses the `concrete_data.csv` dataset which contains several features that influence concrete strength:
- Input features: Various concrete mixture components and environmental factors
- Target variable: Concrete compressive strength ('Strength' column)

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Keras/TensorFlow

You can install the required dependencies with:
```
pip install numpy pandas scikit-learn keras tensorflow
```

## Project Structure
- `RegressionModel_s.py` - Main script that implements and evaluates different model configurations
- `concrete_data.csv` - Dataset containing concrete mixture properties and strength measurements

## Model Configurations
The project evaluates four different model configurations:

### Part A: Baseline Model
- Single hidden layer with 10 nodes
- ReLU activation function
- 50 training epochs
- Raw input features (no normalization)

### Part B: Normalized Features
- Same architecture as baseline
- Input features are normalized (zero mean, unit variance)
- 50 training epochs

### Part C: Extended Training
- Normalized features
- Single hidden layer with 10 nodes
- 100 training epochs (increased from 50)

### Part D: Deeper Network
- Normalized features
- Three hidden layers with 10 nodes each
- 50 training epochs

## Evaluation
Each model configuration is evaluated through 50 trials to account for the random initialization of network weights. Performance is measured using Mean Squared Error (MSE), with both the mean and standard deviation reported across trials.

## Usage
Run the script with:
```
python RegressionModel_s.py
```

The script will automatically:
1. Load and prepare the dataset
2. Train and evaluate each model configuration
3. Output performance metrics for each configuration

## Results
The expected output includes mean and standard deviation of MSE for each configuration:

```
Baseline Mean MSE: [value]
Baseline Std MSE: [value]

Normalized Mean MSE: [value]
Normalized Std MSE: [value]

100 Epochs Mean MSE: [value]
100 Epochs Std MSE: [value]

Three Layers Mean MSE: [value]
Three Layers Std MSE: [value]
```

Lower MSE values indicate better prediction performance.

## Insights
This project demonstrates the following:
- The importance of feature normalization for neural network performance
- How training duration (epochs) affects model accuracy
- The impact of network depth on modeling complex relationships
