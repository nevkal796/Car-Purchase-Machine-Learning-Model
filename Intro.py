# Car Sales Prediction using a Neural Network
# Author: Nevin Kalloor
# Description: This script trains a neural network to predict car purchase amounts based on customer data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------------------------------------
# 1. Load and Inspect the Data
# ---------------------------------------------

# Load dataset with appropriate encoding
car_dat = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

# Optional: Visualize relationships between features (commented to avoid slow rendering)
# sns.pairplot(car_dat)

# ---------------------------------------------
# 2. Preprocess the Data
# ---------------------------------------------

# Separate input features and output target
# Drop irrelevant or non-numeric columns
trim = car_dat.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
out = car_dat['Car Purchase Amount'].values.reshape(-1, 1)

# Initialize separate scalers for input and output
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

# Scale features and labels
trim_scaled = input_scaler.fit_transform(trim)
out_scaled = output_scaler.fit_transform(out)

# Split dataset into training and testing sets
trim_train, trim_test, out_train, out_test = train_test_split(trim_scaled, out_scaled, test_size=0.15)

# ---------------------------------------------
# 3. Build and Train the Neural Network
# ---------------------------------------------

# Define a simple feedforward neural network
model = Sequential()
model.add(Dense(100, input_dim=5, activation='relu'))  # Input layer
model.add(Dense(100, activation='relu'))               # Hidden layer
model.add(Dense(1, activation='linear'))               # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
epochs_hist = model.fit(
    trim_train, out_train,
    epochs=100,
    batch_size=50,
    verbose=1,
    validation_split=0.1
)

# ---------------------------------------------
# 4. Visualize Training Progress
# ---------------------------------------------

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title("Model Loss Progress During Training")
plt.ylabel("Training and Validation Loss")
plt.xlabel("Epoch Number")
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# ---------------------------------------------
# 5. Make a Prediction
# ---------------------------------------------

# New sample data: [Gender, Age, Annual Salary, Credit Card Debt, Net Worth]
sample_input = np.array([[1, 50, 50000, 10000, 600000]])
sample_input_scaled = input_scaler.transform(sample_input)

# Predict and inverse-transform the output to original scale
prediction_scaled = model.predict(sample_input_scaled)
prediction_actual = output_scaler.inverse_transform(prediction_scaled)

# Output the result
print("Predicted Car Purchase Amount: $", prediction_actual[0][0])
