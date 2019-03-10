from keras.layers import Dense, Convolution2D, MaxPooling2D, Input
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

path = './datasets/'

def load_data():
	dataset = []
	filenames = os.listdir(path)
	index = 0
	for file in filenames:
		data = np.loadtxt(path+file, dtype='float32')
		dataset.append(data)
		index = index + 1
		if index % 10 == 0:
			break
			#print(index)
	#print(np.array(dataset).shape)
	return dataset

x_train = load_data()
x_test = x_train[1][:][:]

x_train = np.reshape(x_train, (len(x_train), 30000))
x_test = np.reshape(x_test, (1, 30000))
print(np.array(x_train).shape, np.array(x_test).shape)

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

model = Sequential()
'''model.add(Dense(1000, activation = 'relu', input_dim = 30000))
model.add(Dense(500, activation= 'relu'))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(500, activation= 'relu'))
model.add(Dense(1000, activation= 'relu'))
model.add(Dense(30000, activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')'''

model.add(Dense(700, activation = 'relu', input_dim = 30000))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(700, activation= 'relu'))
model.add(Dense(30000, activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train_noisy, x_train, 
          nb_epoch=100,
          batch_size=30000,
          shuffle=True,
          validation_data=(x_train_noisy, x_train))

'''
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


'''
'''
user1 = np.loadtxt('dataset/random0.txt', dtype='int64')
user2 = np.loadtxt('dataset/random1.txt', dtype='int64')
print(user1, user1.shape, user2, user2.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = user1[:, 0]
y = user1[:, 1]
z = user1[:, 2]
ax.scatter(x, y, z, c = 'r', marker = 'o')

plt.show()'''