import pygame as pg
import random
import os, random, scipy.misc
import numpy as np
from PIL import Image

white = (255, 255, 255)
width = 28
height = 28

def addNoise():
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

path = 'train/'
savepath = 'trains/'
label = ["circle", "square", "triangle", "horizontal", "vertical", "diagonal"]

for j in range(0, 6):
	for i in range(0, 150):
		win = pg.display.set_mode((width, height))
		win.fill(white)
		if j == 0:
			if random.choice([1, -1]) == 1:
				first = random.randint(2, 25)
				sec = random.randint(2, 25)
				third = random.randint(first, 27)
				fourth = random.randint(sec, 27)
				while abs(sec - fourth) <= 7 or abs(first - third) <= 7 or max(first, third)+abs(first-third) > 28 or max(sec, fourth)+abs(sec-fourth) > 28:
					first = random.randint(2, 25)
					sec = random.randint(2, 25)
					third = random.randint(first, 27)
					fourth = random.randint(sec, 27)
				pg.draw.ellipse(win, (0, 0, 0), (first, sec, third, fourth))
			else: 
				center = (random.randint(2, 28), random.randint(2, 28))
				radius = min([random.randint(2, 15), min(center)])
				while (center[0] + radius > 28 or center[1] + radius > 28) and (center[0] - radius < 0 or center[1] - radius < 0):
					center = (random.randint(2, 28), random.randint(2, 28))
				pg.draw.circle(win, (0, 0, 0), center, radius, 0)		
		elif j == 1:
			first = random.randint(1, 25)
			sec = random.randint(1, 25)
			third = random.randint(first, 28)
			fourth = random.randint(sec, 28)
			while abs(sec - fourth) <= 7 or abs(first - third) <= 7 or max(first, third)+abs(first-third) > 28 or max(sec, fourth)+abs(sec-fourth) > 28:
				first = random.randint(1, 25)
				sec = random.randint(1, 25)
				third = random.randint(first, 28)
				fourth = random.randint(sec, 28)
			pg.draw.rect(win, (0, 0, 0), (first, sec, third, fourth))
		elif j == 2:
			top = (random.randint(1, 25), random.randint(1, 23))
			h = random.randint(3, 13)
			w = random.randint(5, 15)
			left = (top[0] - w, top[1] +  h)
			right = (top[0] + w, top[1] + h)
			pg.draw.polygon(win, (0, 0, 0), [top, left, right])
		elif j == 3:
			start = (random.randint(2, 25), random.randint(2, 25))
			end = (random.randint(2, 27), start[1] + random.randint(0, 1)*random.choice([+1, -1])) # random.randint(0, 1)*random.choice([+1, -1])
			while abs(start[0] - end[0]) < 5:
				end = (random.randint(2, 27), start[1] +random.randint(0, 1)*random.choice([+1, -1])) # random.randint(0, 1)*random.choice([+1, -1])
			pg.draw.line(win, (0, 0, 0), start, end, random.randint(1, 3))
		elif j == 4:
			start = (random.randint(2, 25), random.randint(2, 25))
			end = (start[0]+random.randint(0, 1)*random.choice([+1, -1]), random.randint(2, 27))
			while abs(start[1] - end[1]) < 5 :
				end = (start[0]+random.randint(0, 1)*random.choice([+1, -1]), random.randint(2, 27))
			pg.draw.line(win, (0, 0, 0), start, end, random.randint(1, 3))
		elif j == 5:
			start = end = 0
			while True:
				start = (random.randint(1, 28), random.randint(1, 28))
				end = (random.randint(1, 28), random.randint(1, 28))
				if (abs(start[0] - end[0]) >= 8 and abs(start[1] - end[1]) >=8):
					break
			pg.draw.line(win, (0, 0, 0), start, end, random.randint(1, 4))
		fname = path + label[j] + str(i) + ".png"
		pg.image.save(win, fname)
		#pg.display.flip()
pg.quit()

addNoise()