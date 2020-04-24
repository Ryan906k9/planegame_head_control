import pygame
from random import *

class Bomb(pygame.sprite.Sprite):
    def __init__(self,size):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("images/bomb_supply.png").convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.size = size
        self.rect.left, self.rect.top = randint(0, self.size[0]-self.rect.width), -4 * self.rect.height
        self.speed = 2
        self.active = False

    def move(self):
        if self.rect.top < self.size[1] -110:
            self.rect.top += self.speed
            
        else:
            self.active = False

    def reset(self):
        self.active = True
        self.rect.left, self.rect.top = randint(0, self.size[0]-self.rect.width), -4 * self.rect.height

#奖励炸弹
class Bomb1(pygame.sprite.Sprite):
    def __init__(self,size,position):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("images/bomb_supply.png").convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.size = size
        self.rect.left, self.rect.top = position
        self.speed = 5
        self.active = True

    def move(self):
        if self.rect.top < self.size[1] -110:
            self.rect.top += self.speed
            
        else:
            self.active = False
    """
    def reset(self):
        self.active = True
        self.rect.left, self.rect.top = randint(0, self.size[0]-self.rect.width), -4 * self.rect.height
    """


class Bullet(pygame.sprite.Sprite):
    def __init__(self,size):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("images/bullet_supply.png").convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.size = size
        self.rect.left, self.rect.top = randint(0, self.size[0]-self.rect.width), -3 * self.rect.height
        #self.speed = 2
        self.speed = 4
        self.active = False

    def move(self):
        if self.rect.top < self.size[1] -110:
            self.rect.top += self.speed
        else:
            self.active = False

    def reset(self):
        self.active = True
        self.rect.left, self.rect.top = randint(0, self.size[0]-self.rect.width), -3 * self.rect.height
        

class Bullet1(pygame.sprite.Sprite):
    def __init__(self,size,position):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load("images/bullet_supply.png").convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.size = size
        self.rect.left, self.rect.top = position
        self.speed = 4
        self.active = True

    def move(self):
        if self.rect.top < self.size[1] -110:
            self.rect.top += self.speed
        else:
            self.active = False
    """
    def reset(self):
        self.active = True
        self.rect.left, self.rect.top = randint(0, self.size[0]-self.rect.width), -3 * self.rect.height
    """
        
    
