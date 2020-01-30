import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('mydataCOLOR/50.csv')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#y = y[:,:-1]

classifier = Sequential()
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'linear', input_dim = 12))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'linear'))
#classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'linear'))
#classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'linear'))

#classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'linear'))
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'linear'))


classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size =100, epochs = 100)

y_pred = classifier.predict(X_test)
#a = y_pred
#b = np.ones((a.shape[0], 1))
#z = np.concatenate((a,b), axis = 1)
#print(z)
#print(y_pred)
z = y_pred
#print(sc.inverse_transform(z))

p = y_test
a = 0
#print(sc.inverse_transform(p))
mu = 0
t=[]
for x in range(p.shape[0]):
    mu=(math.sqrt(keras.losses.mean_squared_error(p[x,:], z[x,:])))
    a+=mu
    t = np.append(t,mu)
unique, counts = np.unique(np.around(t,decimals=1), return_counts=True)
print(dict(zip(unique,(100*counts)/p.shape[0])))

print(a/p.shape[0])

model_json = classifier.to_json()
with open("model_color.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model_color.h5")
print("Saved model to disk")


a = 0
dataset = pd.read_csv('mydataCOLOR/Fake/8.csv')

X = dataset.iloc[:, :].values
y = dataset.iloc[:, :].values
t = []
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler(
#scy = StandardScaler()
X = sc.transform(X)
y = sc.transform(y)
print()
y_pred = classifier.predict(X)
mo = 0
for x in range(y.shape[0]):
    mo = (math.sqrt(keras.losses.mean_squared_error(y[x,:], y_pred[x,:]))) 
    a+=mo
    t=np.append(t,mo)
#numpy.around(scipy.stats.mode(t, axis=0), decimals=1)
unique, counts = np.unique(np.around(t,decimals=1), return_counts=True)
print(dict(zip(unique,(100*counts)/y.shape[0])))
#print(scipy.stats.mode(np.around(t,decimals=1), axis=0))
print(a/y.shape[0])
