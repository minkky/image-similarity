import pygame as pg
import random

white = (255, 255, 255)
width = 28
height = 28


#pg.display.set_caption("Pygame draw circle and save")

#center = (width//2, height//2)
#radius = min(center)

index = 0
while True:
	if index == 31:
		break
	win = pg.display.set_mode((width, height))
	win.fill(white)
	#pg.draw.line(win, (0, 0, 0), (random.randint(1, 28), random.randint(1, 28)), (random.randint(1, 28), random.randint(1, 28)), random.randint(1, 4))
	'''
	top = (random.randint(1, 28), random.randint(1, 28))
	h = random.randint(1, 15)
	w = random.randint(1, 10)
	left = (top[0] - w, top[1] +  h)
	right = (top[0] + w, top[1] + h)

	pg.draw.polygon(win, (0, 0, 0), [top, left, right])
	'''
	pg.draw.rect(win, (0, 0, 0), (random.randint(1, 28), random.randint(1, 28), random.randint(1, 50), random.randint(1, 15)))
	#center = (random.randint(1, 28), random.randint(1, 28))
	#radius = min([random.randint(1, 15), min(center)])
	#pg.draw.circle(win, (0, 0, 0), center, radius, 0)
	fname = "./make/rec" + str(index) + ".png"
	index = index + 1
	pg.image.save(win, fname)
	pg.display.flip()
pg.quit()
