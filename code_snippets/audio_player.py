import pygame
import time

pygame.mixer.init()

pygame.mixer.music.load("beep.mp3")

for i in range(5):
    time.sleep(0.25)
    pygame.mixer.music.play()
