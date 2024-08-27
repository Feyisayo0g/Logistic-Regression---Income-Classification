# Import necessary libraries for data manipulation, model training, and evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names for the dataset
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Load the dataset into a pandas DataFrame and assign column names
df = pd.read_csv('adult.data', header=None, names=col_names)

# Clean the dataset by stripping extra whitespace from all object-type columns
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()

# Display the first few rows of the cleaned dataset
print(df.head())

# 1. Check for class imbalance in the target variable (income)
# This will show the proportion of each income class (<=50K, >50K)
print(df.income.value_counts(normalize=True))

# 2. Create the feature dataframe X by selecting relevant columns and converting categorical variables to dummy variables
# 'drop_first=True' drops the first category to avoid multicollinearity
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'hours-per-week', 'education']
X = pd.get_dummies(df[feature_cols], drop_first=True)

# 3. Create a heatmap to visualize the correlation between features
# Helps in understanding multicollinearity and feature relationships
plt.figure()
sns.heatmap(X.corr())  # Correlation matrix heatmap
plt.show()
plt.close()

# 4. Create the output variable y, which is binary:
# 0 when income is <=50K, 1 when income is >50K
y = np.where(df.income == '<=50K', 0, 1)

# 5a. Split the data into training and testing sets
# Randomly splits data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=.2)

# 5b. Fit a Logistic Regression model on the training set and predict on the test set
# Using L1 regularization to handle potential multicollinearity and 'liblinear' solver
log_reg = LogisticRegression(C=0.05, penalty='l1', solver='liblinear')
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

# 6. Print the model parameters (intercept and coefficients)
# This helps in understanding the impact of each feature on the prediction
print('Model Parameters, Intercept:')
print(log_reg.intercept_[0])
print('Model Parameters, Coefficients:')
print(log_reg.coef_)

# 7. Evaluate the model predictions on the test set
# Print the confusion matrix and accuracy score to assess model performance
print('Confusion Matrix on test set:')
print(confusion_matrix(y_test, y_pred))
print(f'Accuracy Score on test set: {log_reg.score(x_test, y_test)}')

# 8. Create a DataFrame of the model coefficients and corresponding feature names
# Sort the coefficients to identify the most and least important features
coef_df = pd.DataFrame(zip(x_train.columns, log_reg.coef_[0]), columns=['Feature', 'Coefficient']).sort_values('Coefficient')
coef_df = coef_df[coef_df.Coefficient.abs() > 0].sort_values('Coefficient')
print(coef_df)

# 9. Plot a barplot of the coefficients sorted in ascending order
# Visual representation of feature importance
plt.figure(figsize=(10,7))
sns.barplot(data=coef_df, x='Feature', y='Coefficient')
plt.xticks(rotation=90)
plt.title('Logistic Regression Coefficient Values')
plt.show()
plt.close()

# 10. Plot the ROC curve and print the AUC value
# The ROC curve helps in evaluating the performance of the classification model
y_pred_prob = log_reg.predict_proba(x_test)
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
print(f'ROC AUC score: {roc_auc}')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0,1],[0,1], color='navy', linestyle='--')
plt.title('ROC Curve')
plt.grid()
plt.show()
