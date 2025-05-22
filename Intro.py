import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


car_dat = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')
#sns.pairplot(car_dat)
trim = car_dat.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'], axis=1)
out = car_dat['Car Purchase Amount']

input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

# Fit and transform inputs and outputs separately
trim_scaled = input_scaler.fit_transform(trim)

out = out.values.reshape(-1,1)
out_scaled = output_scaler.fit_transform(out)

trim_train,trim_test, out_train,out_test = train_test_split(trim_scaled,out_scaled,test_size=.15)

model = Sequential()
model.add(Dense(100, input_dim = 5, activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
epochs_hist = model.fit(trim_train,out_train, epochs = 100, batch_size = 50, verbose = 1, validation_split = 0.1)

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title("Model Loss Progress During Training")
plt.ylabel("Training and Validation Loss")
plt.xlabel("Epoch Number")
plt.legend(['Training Loss','Validation Loss'])
plt.show()

#Gender, Age, Annual Salary, Credit Card Debt, Net Worth
trim_test = np.array([[1,50,50000,10000,600000]])
trim_test_scaled = input_scaler.transform(trim_test)
out_pre = model.predict(trim_test_scaled)
out_original = output_scaler.inverse_transform(out_pre)
print(out_original)
