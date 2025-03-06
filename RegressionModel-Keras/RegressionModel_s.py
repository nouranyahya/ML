import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.layers import Dense, Input

# Load data
print("Loading data...")
data = pd.read_csv('/Users/nouranhussain/ML/RegressionModel-Keras/concrete_data.csv')
X = data.drop('Strength', axis=1) #take all except strength column
y = data['Strength'] #target column

def run_model(X, y, epochs=50, n_layers=1):

    # Split data - 30% testing, 70% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Initialize model with input layer and first hidden layer
    model = Sequential([
        Input(shape=(X.shape[1],)), #input layer
        Dense(10, activation='relu') #first hidden layer
    ])
    
    # Add additional layers if needed - each has 10 nodes
    for _ in range(n_layers - 1):
        model.add(Dense(10, activation='relu'))
    
    # Add output layer with single node
    model.add(Dense(1))
    
    # Compile and train
    model.compile(optimizer='adam', loss='mean_squared_error') #adam optimization algorithm, MSE for regression
    model.fit(X_train, y_train, epochs=epochs, verbose=0) #verbose=0 to suppress output
    
    # Evaluate
    y_pred = model.predict(X_test, verbose=0) #generate predictions
    return mean_squared_error(y_test, y_pred) #calculate MSE

# Part A: Baseline
print("\nPart A: Running baseline model...")
mse_scores_a = [] #list to store scores
for i in range(50): #50 trials
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/50 trials") #prints progress every 10 trials
    mse = run_model(X, y) #train and evaluate model
    mse_scores_a.append(mse) #store score
print(f"Baseline Mean MSE: {np.mean(mse_scores_a):.2f}")
print(f"Baseline Std MSE: {np.std(mse_scores_a):.2f}")

# Part B: Normalized
print("\nPart B: Running normalized model...")
X_norm = (X - X.mean()) / X.std() #normalize features (subtract mean, divide by std)
mse_scores_b = [] #list storing scores
for i in range(50):
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/50 trials")
    mse = run_model(X_norm, y) #train and evaluate
    mse_scores_b.append(mse)
print(f"Normalized Mean MSE: {np.mean(mse_scores_b):.2f}")
print(f"Normalized Std MSE: {np.std(mse_scores_b):.2f}")

# Part C: More epochs
print("\nPart C: Running with 100 epochs...")
mse_scores_c = []
for i in range(50):
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/50 trials")
    mse = run_model(X_norm, y, epochs=100) #using 100 epochs instead of 50
    mse_scores_c.append(mse)
print(f"100 Epochs Mean MSE: {np.mean(mse_scores_c):.2f}")
print(f"100 Epochs Std MSE: {np.std(mse_scores_c):.2f}")

# Part D: Three layers
print("\nPart D: Running with three layers...")
mse_scores_d = []
for i in range(50):
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/50 trials")
    mse = run_model(X_norm, y, n_layers=3) #trains with three hidden layers
    mse_scores_d.append(mse)
print(f"Three Layers Mean MSE: {np.mean(mse_scores_d):.2f}")
print(f"Three Layers Std MSE: {np.std(mse_scores_d):.2f}")