import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

def train():
  (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
  #стандартизация данных
  mean = x_train.mean(axis=0)
  std = x_train.std(axis=0)

  x_train-=mean
  x_train/=std
  x_test-=mean
  x_test/=std

  model = Sequential()
  model.add(Dense(128, activation="relu", input_shape=(x_train.shape[1],)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
  #mse, mae = model.evaluate(x_test, y_test, verbose=0)
  #print(mae)
  model.save('boston_houses.h5')

#train()

model = load_model('boston_houses.h5')
pred = model.predict(x_test)
print(pred[100][0], y_test[100])
