Required steps to test the model

Open Google Colab and run these codes first. I am numbering this code so that you can run them one by one...


1...


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


2...


def scaling(dataset):
  dataset.iloc[:,0]
  del dataset[dataset.columns[0]]

  cols_to_scale = ['weather','PM25_Concentration','wind_direction','temperature','pressure','humidity','PM10_Concentration','NO2_Concentration','CO_Concentration','O3_Concentration','SO2_Concentration']

  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  dataset[cols_to_scale] = scaler.fit_transform(dataset[cols_to_scale])
  return dataset



3...


def splitting(dataset,win):
  window_size = win

#-------------------------TrainSet---------------------------------

# Initialize empty lists to store X and Y
  X_sequences = []
  Y_values = []

# Iterate through the DataFrame to create sequences
  for i in range(len(dataset) - window_size):
      X_seq = dataset.iloc[i:i+window_size].values
      Y_val = dataset.iloc[i+window_size]['PM25_Concentration']
      X_sequences.append(X_seq)
      Y_values.append(Y_val)

# Convert the lists to NumPy arrays for modeling
  X_train = np.array(X_sequences)
  y_train = np.array(Y_values)

  return X_train,y_train



4...

Then, upload models for any step size and city per your need in Google Colab. I have saved every model named modelB1. Here, B and 1 in modelB1 indicate that the model is trained on the city B dataset, considering step size 1. So select any model with step sizes 1,7,14,30,60 or with different cities B,G,S,T from the file that I have provided.




5...


modelloaded = tf.keras.models.load_model('modelB1.h5')


Inside the load_model, you can give the uploaded model's name or paste the model's path...




6...


df_test = pd.read_csv("/content/drive/MyDrive/Multi-step forecasting in multivariate time series data/B_test.csv")
df_test.sample(5)



Here inside pd.read_csv, you can paste the path of the test dataset or name of the uploaded test dataset..



7...



df_test=scaling(df_test)



8...


X_test,y_test=splitting(df_test,1)  





# Here 1 is the step size, I have taken 1 because I have used modelB1.h5 in case of other model using other step size like 1,7,14,30,60
#so choose step size as per model name..


9...


y_pred=modelloaded.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print(f"MSE: {mse}, MAE: {mae}")




..................................................................


