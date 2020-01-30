import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
dataset = pd.read_csv('mydata/end.csv')

X = dataset.iloc[:, :].values
y = dataset.iloc[:, :].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scy = StandardScaler()
X = sc.fit_transform(X)
y = sc.transform(y)
#y = y[:,:-1]
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.99, random_state = 0)
#y = y[:,:-1]

classifier = Sequential()
classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'linear', input_dim = 13))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
#classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'linear'))
#classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'linear'))

#classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'linear'))
classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'linear'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, y, batch_size = 100, epochs = 1000)

y_pred = classifier.predict(X_test)
#a = y_pred
#b = np.ones((a.shape[0], 1))
#z = np.concatenate((a,b), axis = 1)
#print(z)
#print(y_pred)
z = y_pred
print(sc.inverse_transform(z))

p = y_test

print(sc.inverse_transform(p))
print(math.sqrt(keras.losses.mean_squared_error(p[0,:], z[0,:])))


model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

classifier.save_weights("model.h5")
print("Saved model to disk")