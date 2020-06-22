# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:23:20 2020

@author: 841175
"""

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as lg
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection as model_selection
from keras import models
from keras import layers


def predict(weights, x) :     
    return np.dot(weights, x) 

def displayPoints(x,y,w):
    fig = plt.figure()
    X = x[:,1]
    ax = fig.add_axes([0.1,0.2,0.8,0.7])
    ax.scatter(X,y)

    Xmin = np.min(X)
    Ymin = w[0]*1 + w[1]*Xmin
    Xmax = np.max(X)
    Ymax = w[0]*1 + w[1]*Xmax
    
    Xlist = [Xmin, Xmax]
    Ylist = [Ymin, Ymax]
    
    ax.plot(Xlist, Ylist, color='pink')

    ax.set_title("Suicide Dataset")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

def calcCost(X,W,Y):
    P = np.dot(X,W)
    py = (P - Y)**2
    pysum = np.sum(py)
    
    return pysum/len(X)

def createDataFromScale():
    data = np.loadtxt('./suicideData.csv', delimiter = ',',skiprows = 1,usecols = (0,11,3,1,2,4,5,9,10,6),dtype=np.str)
    X = data[:,0:9]
    Y = data[:,-1]
    
    data[:,4][data[:,4]=="female"] = 1
    data[:,4][data[:,4]=="male"] = 0
    
# =============================================================================
# one hot encoding the generations and countries columns
# =============================================================================
    ohe = OneHotEncoder(categories = 'auto')
    categoricaldata = X[:,0:3]
    changeddata = ohe.fit_transform(categoricaldata).toarray()
    X = np.column_stack((changeddata, X[:,3:]))
    X = X.astype(np.float)
    
# =============================================================================
# standardize
# =============================================================================
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    xStd = (X-mean)/std
    bias = np.ones((len(X),1))
    
    x = np.column_stack((bias, xStd))
# =============================================================================
# make test data
# =============================================================================
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, Y, train_size=0.75,test_size=0.25, random_state=101)

# =============================================================================
# printing weights
# =============================================================================
    print("Country: -.653046 ")
    print("Generation: -4.37503")
    print("Age: 1.27469")
    print("Year: 4.10526 ")
    print("Gender: -0.54222 ")
    print("#ofSuicide: -6.68798 ")
    print("Population: 4.85271")
    print("GDP Per Year: -2.97147")
    print("GDP Per Capita: 0.460667")
    print("Suicides/100k: -1.78067")

    return x,Y,mean,std,x_test,y_test,x_train,y_train

def gradientDescent(X,Y,W) :
    s = 0.0
    pMatrix = np.dot(X,W)
    pyMatrix = pMatrix - Y
    s = np.dot(X.T, pyMatrix)
    return s/len(X)
    
# =============================================================================
#                             end of functions
# =============================================================================

X,Y,mean,std,xtest,ytest,xtrain,ytrain = createDataFromScale()
Y = Y.astype(np.float)


listOfLR = [0.02]
fig = plt.figure()     
ax = fig.add_axes([0.1,0.1,0.8,0.8])# [left, bottom, width, height]
for lr in listOfLR :
    
    W=[0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
    C = calcCost(X,W,Y)
    costArray = []
    costArray.append(calcCost(X, W, Y))
    
    finish = False
    count = 0
    while not finish and count < 10000:
          gradientList = gradientDescent(X,Y,W)
          W = W - lr*(gradientList)
          gradMagnitude = lg.norm(gradientList)
          count+=1

          costArray.append(calcCost(X, W, Y))
    
          if gradMagnitude < 0.001 :
              finish = True
         
    
    ax.plot(np.arange(len(costArray)), costArray,label = lr)
displayPoints(X,Y,W)
ax.legend()
print("ending cost = ", calcCost(X,W,Y))
print("initial cost = ", C)

# =============================================================================
# Keras Plotting
# =============================================================================

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape = (len(X[0]),) ))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(xtrain, ytrain, validation_split=0.25, epochs=4, batch_size=1, verbose=1)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
mean_absolute_error = history.history['mean_absolute_error']
val_mean_absolute_error = history.history['val_mean_absolute_error']

# =============================================================================
# Fig 1
# =============================================================================
plt.figure(1)
plt.title('Model Loss')
plt.plot(epochs,loss,'r-',label='Training Loss')
plt.plot(epochs,val_loss,'b-',label='Validation Loss')
plt.xlabel('epochs')
plt.legend(['train','validation'],loc='upper right')
plt.grid()
plt.show()

# =============================================================================
# Fig 2
# =============================================================================
plt.figure(2)
plt.title('Model MAE')
plt.plot(epochs,mean_absolute_error,'r-',label='Training MAE')
plt.plot(epochs,val_mean_absolute_error,'b-',label='Validation MAE')
plt.xlabel('epochs')
plt.legend(['train','validation'],loc='upper right')
plt.grid()
plt.show()
