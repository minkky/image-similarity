import os, random, scipy.misc
import numpy as np
from PIL import Image

path = 'train/'
savepath = 'trains/'

def load_data():
	dataset = []
	filenames = os.listdir(path)
	
	for file in filenames:
		print(file)
		pic = Image.open(path + file)
		pic = np.array(pic)[:, :, 1]
		#print(pic)
		index = 0
		picture = []
		for p in pic:
			pics = []
			for i in p:
				if i == 0:
					pics.append(random.randint(0, 150))
				elif i == 255:
					pics.append(random.randint(150, 255))
			picture.append(pics)
		picture = np.array(picture)
		scipy.misc.imsave(savepath+file, picture)

load_data()