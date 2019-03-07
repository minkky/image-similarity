import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

user1 = np.loadtxt('dataset/random0.txt', dtype='int64')
user2 = np.loadtxt('dataset/random1.txt', dtype='int64')
print(user1, user1.shape, user2, user2.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = user1[:, 0]
y = user1[:, 1]
z = user1[:, 2]
ax.scatter(x, y, z, c = 'r', marker = 'o')

plt.show()