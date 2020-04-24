import pygame
import sys
class Plane(pygame.sprite.Sprite):
    def __init__(self,size):
        pygame.sprite.Sprite.__init__(self)
        self.image1 = pygame.image.load("images/me1.png").convert_alpha()
        self.image2 = pygame.image.load("images/me2.png").convert_alpha()
        self.rect = self.image1.get_rect()
        self.size = size
        self.rect.left, self.rect.top = (self.size[0] - self.rect.width)//2, self.size[1]-self.rect.height-57        
        self.speed = 10
        self.destroy_image = []
        self.destroy_image.extend([\
            pygame.image.load("images/me_destroy_1.png").convert_alpha(),\
            pygame.image.load("images/me_destroy_2.png").convert_alpha(),\
            pygame.image.load("images/me_destroy_3.png").convert_alpha(),\
            pygame.image.load("images/me_destroy_4.png").convert_alpha()])
        self.active = True
        self.invincible = False
        self.blink = False
        self.mask = pygame.mask.from_surface(self.image1)

    def move_up(self):
        self.rect.top -= self.speed
        if self.rect.top < 0:
            self.rect.top = 0
            
    def move_down(self):
        self.rect.top += self.speed
        if self.rect.bottom > self.size[1] - 57 :
            self.rect.bottom = self.size[1]- 57

    def move_left(self):
        self.rect.left -= self.speed
        if self.rect.left < 0:
            self.rect.left = 0
            
    def move_right(self):
        self.rect.left += self.speed
        if self.rect.right > self.size[0]:
            self.rect.right = self.size[0]

    def reset(self):
        self.active = True
        self.invincible = True
        self.rect.left, self.rect.top = (self.size[0] - self.rect.width)//2, self.size[1]-self.rect.height-57
