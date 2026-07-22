🚗 Car Sales Amount Prediction with Neural Networks
This project uses a deep learning model to predict the car purchase amount for customers based on demographic and financial data.
📊 Dataset
The dataset includes the following columns:
Customer Name
Customer e-mail
Country
Gender
Age
Annual Salary
Credit Card Debt
Net Worth
Car Purchase Amount (Target)
500 records, encoded using ISO-8859-1. Irrelevant identifying columns (Customer Name, Customer e-mail, Country) are dropped before training, leaving 5 numeric input features.
🧠 Model Overview
The neural network is built using TensorFlow and Keras and consists of:
Input Layer with 5 features
Two hidden layers with 100 neurons each, using ReLU activation
Output Layer with linear activation to predict purchase amount
Inputs and outputs are each scaled independently with MinMaxScaler before training, and the output scaler is used to invert predictions back to real dollar values. The model is trained using Mean Squared Error loss and the Adam optimizer, over 100 epochs with a 10% validation split and an 85/15 train/test split.
📈 Results
Evaluated on the held-out test set (inverse-transformed back to real dollar values):
Test MAPE: 0.33%
Test MAE: ~$139 (on purchase amounts averaging roughly $44,000)
Baseline comparison: predicting the test-set mean for every sample gives a MAPE of ~21% and MAE of ~$8,444 — so the trained model reduces error by roughly 98% over that baseline
Training and validation loss (MSE) were tracked across epochs to confirm the model converged without overfitting
📈 Training Visualization
After training, a loss curve is plotted to visualize training vs. validation performance over epochs.
🔍 Prediction Example
The model predicts the car purchase amount for a sample input:
[Gender, Age, Annual Salary, Credit Card Debt, Net Worth]
[1, 50, 50000, 10000, 600000]
