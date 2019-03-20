from keras.layers import Dense, Convolution2D, MaxPooling2D, Input, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.preprocessing import MinMaxScaler

path = './test300/'

def load_data():
	dataset = []
	filenames = os.listdir(path)
	index = 0
	for file in filenames:
		data = np.loadtxt(path+file, dtype='float32')
		dataset.append(data)
		index = index + 1
	print(np.array(dataset).shape)
	return dataset

x_train = load_data()
x_train = np.array(x_train)

x_test = x_train[1:15][:][:]
x_test = np.array(x_test)

model = Sequential()
encoded = LSTM(256, input_shape=(299, 3), activation='relu', return_sequences = True)
model.add(encoded)
model.add(Dropout(0.5))
model.add(LSTM(128, activation='relu', return_sequences = True))
model.add(LSTM(64, activation='relu', return_sequences = True))
model.add(LSTM(32, activation='relu', return_sequences = True))

#model.add(Dense(1))
#model.add(Activation('relu'))
decoded = LSTM(3, return_sequences = True)
model.add(decoded)

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train, x_train, 
          nb_epoch=100,
          batch_size=100,
          shuffle=False,
          validation_data=(x_test, x_test))

res = model.predict(x_test)
print(x_test, res)
