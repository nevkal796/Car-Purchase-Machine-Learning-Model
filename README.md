# 🚗 Car Sales Amount Prediction with Neural Networks

This project uses a deep learning model to predict the **car purchase amount** for customers based on demographic and financial data.

---

## 📊 Dataset

The dataset includes the following columns:

- `Customer Name`
- `Customer e-mail`
- `Country`
- `Gender`
- `Age`
- `Annual Salary`
- `Credit Card Debt`
- `Net Worth`
- `Car Purchase Amount` (Target)

> The dataset is assumed to be in CSV format and encoded using ISO-8859-1.

---

## 🧠 Model Overview

The neural network is built using TensorFlow and Keras and consists of:

- Input Layer with 5 features
- Two hidden layers with 100 neurons each, using ReLU activation
- Output Layer with linear activation to predict purchase amount

The model is trained using **Mean Squared Error** loss and the **Adam** optimizer.

---

## 📈 Training Visualization

After training, a loss curve is plotted to visualize training vs. validation performance over epochs.

---

## 🔍 Prediction Example

The model predicts the car purchase amount for a sample input:
```python
[Gender, Age, Annual Salary, Credit Card Debt, Net Worth]
[1, 50, 50000, 10000, 600000]


