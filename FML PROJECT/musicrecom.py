import sys
import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense ,Dropout,BatchNormalization
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
train_X = pd.read_csv("DataSet/muse_v3.csv", delimiter=',')
train_Y=train_X['lastfm_url']
train_data=train_X[['valence_tags','arousal_tags','dominance_tags']]
def baseline_model():
    # create model
    model = Sequential()
    
    
    
    model.add(Dense(16, input_dim=3, activation='relu'))
    
    
    model.add(Dense(8, input_dim=3, activation='relu'))
    
    
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
Y=np.arange(0, 90001, 1, dtype=int)
X_train, X_test, y_train, y_test = train_test_split(
     np.array(train_data),Y, test_size=0.2, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
estimator = KerasRegressor(build_fn=baseline_model, epochs=30, batch_size=3, verbose=1)
history=estimator.fit(X_train,y_train)
plt.plot(history.history['loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
estimator.model.save('drive/My Drive/FML_Assignment/model_weights.h5')

with open('drive/My Drive/FML_Assignment/model_architecture.json', 'w') as f:
    f.write(estimator.model.to_json())