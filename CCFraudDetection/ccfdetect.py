#Credit Card Fraud Detection with Decision Trees and SVM

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import LinearSVC

# Load data
print("Loading data...")
rawdata = pd.read_csv('/Users/nouranhussain/ML/CCFraudDetection/creditcard.csv')

# Plot 1: Class distribution pie chart
labels = rawdata.Class.unique() # get the set of distinct classes
sizes = rawdata.Class.value_counts().values # get the count of each class

# plot the class value counts
#0 (the credit card transaction is legitimate) and 1 (the credit card transaction is fraudulent)
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.savefig('class_distribution.png')
plt.show()

# Plot 2: Feature correlation with Class
plt.figure(figsize=(10, 6))
correlation_values = rawdata.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh')
plt.title('Feature Correlation with Class')
plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.show()

# standardize features by removing the mean and scaling to unit variance
rawdata.iloc[:, 1:30] = StandardScaler().fit_transform(rawdata.iloc[:, 1:30])
data_matrix = rawdata.values

X = data_matrix[:, 1:30] # X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
y = data_matrix[:, 30] # y: labels vector

X = normalize(X, norm="l1") # data normalization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
w_train = compute_sample_weight('balanced', y_train) #compute sample weights to take into account class imbalance in dataset

# Train Decision Tree model
dt = DecisionTreeClassifier(max_depth=4, random_state=35) #for reproducible output across multiple function calls, set random_state to a given integer value
dt.fit(X_train, y_train, sample_weight=w_train)

# Train SVM model
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False) #for reproducible output across multiple function calls, set random_state to a given integer value
svm.fit(X_train, y_train)

# Decision Tree predictions
y_pred_dt = dt.predict_proba(X_test)[:,1] #compute probability of belonging to fraudulent transactions class
roc_auc_dt = roc_auc_score(y_test, y_pred_dt) #AUC-ROC score evaluates your model's ability to distinguish positive and negative classes considering all possible probability thresholds
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

# SVM predictions
y_pred_svm = svm.decision_function(X_test) #compute probability of belonging to fraudulent transactions class
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

# Plot 3: ROC curves for both models
plt.figure(figsize=(10, 8))
# ROC Curve for Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.3f})')

# ROC Curve for SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.3f})')

# Diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.savefig('roc_curves.png')
plt.show()

# Plot 4: Top correlated features
plt.figure(figsize=(10, 6))
correlation_values = abs(rawdata.corr()['Class']).drop('Class')
top_corr = correlation_values.sort_values(ascending=False)[:6]
print("Top 6 correlated features:")
print(top_corr)
top_corr.plot(kind='barh')
plt.title('Top 6 Features by Correlation with Class')
plt.tight_layout()
plt.savefig('top_features.png')
plt.show()