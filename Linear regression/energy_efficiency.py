# Energy Efficiency - Linear Regression Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_excel('ENB2012_data.xlsx')
print("Dataset loaded successfully")
print(df.head())

# Rename columns
df.columns = ['Relative Compactness',
              'Surface Area',
              'Wall Area',
              'Roof Area',
              'Overall Height',
              'Orientation',
              'Glazing Area',
              'Glazing Area Distribution',
              'Heating Load',
              'Cooling Load']

# Separate features and target
data = df.iloc[:, 0:8]
target = df.iloc[:, 8]

print("\nFeatures shape:", data.shape)
print("Target shape:", target.shape)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("\nTraining set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Train the Linear Regression model
regression = LinearRegression()
regression.fit(x_train, y_train)

print("\nModel trained successfully")

# Perform cross-validation
mse = cross_val_score(regression, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
print(f"\nCross-validation MSE: {np.mean(mse):.4f}")

# Make predictions on test set
x_predict = regression.predict(x_test)

print("\nPredictions completed")
print(f"Number of predictions: {len(x_predict)}")

# Visualize the residuals
sns.displot(x_predict - y_test)
plt.xlabel('Residuals (Prediction - Actual)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

print("\nModel evaluation complete")
