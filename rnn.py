from keras.layers import Dense, Convolution2D, MaxPooling2D, Input, SimpleRNN
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

path = 'datasets/'

def load_data():
	dataset = []
	filenames = os.listdir(path)
	index = 0
	for file in filenames:
		data = np.loadtxt(path + file, dtype='float32')
		dataset.append(data)
		index = index + 1
		'''
		if index % 10 == 0 :
			break
			#print(index)
		'''
	print(np.array(dataset).shape)
	return dataset

input_data = load_data()

np.random.seed(0)
model = Sequential()
model.add(SimpleRNN(5000, input_shape=(10000, 3)))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='sgd')

model.fit(input_data)
