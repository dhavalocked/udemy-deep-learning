# -*- coding: utf-8 -*-

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values 
Y = dataset.iloc[:,13].values 

# Encode the categorial data columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X_1 = LabelEncoder()
X[:, 1] = LabelEncoder_X_1.fit_transform(X[:,1])
LabelEncoder_X_2 = LabelEncoder()
X[:, 2] = LabelEncoder_X_2.fit_transform(X[:,2])

# encode the categorial features
onehotencoder = OneHotEncoder(categorical_features=[1])
X= onehotencoder.fit_transform(X).toarray()

# remove one columns to avoid dummy variable trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2, random_state = 0)

# Scale or normalize
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import keras libraries and other packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize ANN
classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=11, units=6))
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=6))

# Adding the output layer
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

#Compile the model
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size=10, epochs=100 )

#predict the results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_tesy, Y_pred)











