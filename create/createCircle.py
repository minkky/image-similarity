import random, os
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from glob import glob
from PIL import Image
import numpy as np

IMG_WIDTH = 28
IMG_HEIGHT = 28
path = '../dataset/'
filename = 'circle'
index = 0
#LEFTUP = 0, LEFTDOWN = 1, CENTER = 2, RIGHTUP = 3, RIGHTDOWN = 4
#loc = [LEFTUP, LEFTDOWN, CENTER, RIGHTUP, RIGHTDOWN]

'''
def selectLoc():
	return loc[random.randint(0, 4)]
location = selectLoc()
width = random.randint(3, 8)
height = random.randint(3, 8)
'''

def load_data():
	dataset = []
	filenames = os.listdir(path)
	index = 0
	for file in filenames:
		pic = Image.open(path + file)
		pic = np.array(pic)[:, :, 1]
		dataset.append(pic)
		'''
		index = index + 1
		if index == 1:
			break
			#print(index)'''
	print(np.array(dataset).shape)
	return dataset

dataset = load_data()
