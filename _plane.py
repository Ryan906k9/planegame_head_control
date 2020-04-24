"""飞机生命补给"""
import pygame
from random import *
import math

class Life(pygame.sprite.Sprite):
    def __init__(self,size,position):
        pygame.sprite.Sprite.__init__(self)
        
        self.size = size
        self.image = pygame.image.load("images/life1.png").convert_alpha()
        
        self.mask = pygame.mask.from_surface(self.image)
        self.rect_o = self.image.get_rect()
        #缩小图片
        self.image_list = []
        self.image = pygame.transform.smoothscale(self.image,\
                                                  (int(self.rect_o.width * 0.5),\
                                                   int(self.rect_o.height * 0.5)))
        self.image_list.extend((self.image, \
                                pygame.transform.rotate(self.image,90),\
                                pygame.transform.rotate(self.image,180),\
                                pygame.transform.rotate(self.image,270)))
        
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        self.speed = [randint(0,1), randint(0,1)]
        self.direction = [choice([1,-1]),choice([1,-1])]
        """
        if self.direction[1] * self.speed[1] < 0:
            self.ang = math.atan(self.speed[0]*self.direction[0]/(self.speed[1]*self.direction[1]))/\
                       math.pi* 180
        elif self.direction[1] * self.speed[1] > 0:
            self.ang = math.atan(self.speed[0]*self.direction[0]/(self.speed[1]*self.direction[1]))/\
                       math.pi* 180 + 180
        else:
            if self.direction[0] < 0:
                self.ang = 90
            elif self.direction[0] > 0:
                self.ang = -90
            else:
                self.ang = 0
        self.image = pygame.transform.rotate(self.image,self.ang)
        """
        self.active = True

    def move(self):
        self.rect.left += self.speed[0] * self.direction[0]
        self.rect.top += self.speed[1] * self.direction[1]
        if self.rect.left < 0 or self.rect.right > self.size[0]:
            self.active = False
        if self.rect.top  < 0 or self.rect.bottom > self.size[1]:
            self.active = False
            
