import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
dataset = pd.read_csv('1291.csv')

X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 3:4].values
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scy = StandardScaler()
X = sc.fit_transform(X)
y = sc.transform(y)
"""
print(X)
print(y)
#y = y[:,:-1]
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.99, random_state = 0)
#y = y[:,:-1]

classifier = Sequential()
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'linear', input_dim = 3))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
#classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'linear'))
#classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'linear'))

#classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'linear'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, y, batch_size = 10, epochs = 1000)
print(classifier.summary())
y_pred = classifier.predict(X_test)

model_json = classifier.to_json()
with open("modelNEW.json", "w") as json_file:
    json_file.write(model_json)

classifier.save_weights("modelNEW.h5")
print("Saved model to disk")
print(classifier.get_weights())
