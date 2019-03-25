import pygame

pygame.init()

size = [28, 28]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("circle")

done = False
clock = pygame.time.Clock()

while not done:
	clock.tick(10)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			done = True

		screen.fill((255, 255, 255, 255))
		pygame.draw.circle(screen, (255, 0,0),[10, 20], 10)

		test = pygame.display.flip()
		pygame.image.save(test, "test.png")
pygame.quit()