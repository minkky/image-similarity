import random 

CIRCLE = 0# 동그라미
RECTANGLE = 1# 네모
TRIANGLE = 2# 세모
VERTICAL = 3# 세로 선
DIAGONAL = 4# 대각 선
HORIZONTAL = 5# 가로 선
shape = [CIRCLE, RECTANGLE, TRIANGLE, VERTICAL, DIAGONAL, HORIZONTAL]

IMG_WIDTH = 28
IMG_HEIGHT = 28

def select_shape():
	return shape[random.randint(0, 5)]

def getSize(selectedShape):
	if selectedShape <= 2:
		width = random.randint(3, 10)
		height = random.randint(3, 10)
	else:
		if selectedShape != VERTICAL:
			width = IMG_WIDTH
			if selectedShape == DIAGONAL:
				height = IMG_HEIGHT
			else:
				height = random.randint(1, 5)
		else:
			width = random.randint(1, 5)
			height = IMG_HEIGHT
	return width, height

def getStartPoint():
	
	selectedShape = select_shape()
width, height = getSize(selectedShape)

print(selectedShape, width, height)
