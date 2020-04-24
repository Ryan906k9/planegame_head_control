import pygame
from random import *

class SmallEnemy(pygame.sprite.Sprite):
    def __init__(self,size):
         pygame.sprite.Sprite.__init__(self)
         self.image = pygame.image.load("images/enemy1.png").convert_alpha()
         self.rect = self.image.get_rect()
         self.size = size
         #self.speed = 2
         self.speed = 10
         self.rect.top, self.rect.left = randint(-30 * self.rect.height, 0), \
                                         randint(0,self.size[0]-self.rect.width)
         self.active = True
         self.mask = pygame.mask.from_surface(self.image)
         self.destroy_image = []
         self.destroy_image.extend([\
            pygame.image.load("images/enemy1_down1.png").convert_alpha(),\
            pygame.image.load("images/enemy1_down2.png").convert_alpha(),\
            pygame.image.load("images/enemy1_down3.png").convert_alpha(),\
            pygame.image.load("images/enemy1_down4.png").convert_alpha()])

    def move(self):
        if self.rect.top < self.size[1] -110 :
            self.rect.top += self.speed
        else:
            self.reset()

    def reset(self):
        self.active = True
        self.rect.top, self.rect.left = randint(-25 * self.rect.height, 0), \
                                        randint(0,self.size[0]-self.rect.width)

class MidEnemy(pygame.sprite.Sprite):
    energy = 10
    
    def __init__(self,size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("images/enemy2.png")
        self.image_hit = pygame.image.load("images/enemy2_hit.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.size = size
        self.speed = 1
        self.rect.top, self.rect.left = randint(-35 * self.rect.height, -5 * self.rect.height), \
                                        randint(0, self.size[0]-self.rect.width)
        self.active = True
        self.mask = pygame.mask.from_surface(self.image)
        self.destroy_image = []
        self.destroy_image.extend([\
            pygame.image.load("images/enemy2_down1.png").convert_alpha(),\
            pygame.image.load("images/enemy2_down2.png").convert_alpha(),\
            pygame.image.load("images/enemy2_down3.png").convert_alpha(),\
            pygame.image.load("images/enemy2_down4.png").convert_alpha()])
        self.energy = MidEnemy.energy
        self.hit = False


    def move(self):
        if self.rect.top < self.size[1] -110:
            self.rect.top += self.speed
        else:
            self.reset()

    def reset(self):
        self.energy = MidEnemy.energy
        self.active = True      
        self.rect.top, self.rect.left = randint(-35 * self.rect.height, -5 * self.rect.height), \
                                        randint(0, self.size[0]-self.rect.width)

class BigEnemy(pygame.sprite.Sprite):
    energy = 50
    
    def __init__(self,size):
        pygame.sprite.Sprite.__init__(self)
        self.image1 = pygame.image.load("images/enemy3_n1.png").convert_alpha()
        self.image2 = pygame.image.load("images/enemy3_n2.png").convert_alpha()
        self.image_hit = pygame.image.load("images/enemy3_hit.png").convert_alpha()
        self.rect = self.image1.get_rect()
        self.size = size
        self.speed = 1
        self.rect.top, self.rect.left = randint(-40 * self.rect.height, -5 * self.rect.height), \
                                        randint(0, self.size[0]-self.rect.width)
        self.active = True
        self.mask = pygame.mask.from_surface(self.image1)
        self.destroy_image = []
        self.destroy_image.extend([\
            pygame.image.load("images/enemy3_down1.png").convert_alpha(),\
            pygame.image.load("images/enemy3_down2.png").convert_alpha(),\
            pygame.image.load("images/enemy3_down3.png").convert_alpha(),\
            pygame.image.load("images/enemy3_down4.png").convert_alpha(),\
            pygame.image.load("images/enemy3_down5.png").convert_alpha(),\
            pygame.image.load("images/enemy3_down6.png").convert_alpha()])
        self.energy = BigEnemy.energy
        self.hit = False


    def move(self):
        if self.rect.top < self.size[1] -110:
            self.rect.top += self.speed
        else:
            self.reset()

    def reset(self):
        self.energy = BigEnemy.energy
        self.active = True        
        self.rect.top, self.rect.left = randint(-40 * self.rect.height, -5 * self.rect.height), \
                                        randint(0, self.size[0]-self.rect.width)

class Boss(pygame.sprite.Sprite):
    energy = 200
    def __init__(self,size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("boss/lv1.png").convert_alpha()
        self.image_hit = pygame.image.load("boss/lv1_hit.png").convert_alpha()
        
        self.size = size
        self.rect = self.image.get_rect()
        self.rect.top, self.rect.left = (-1 *self.rect.height), (self.size[0] - self.rect.width)//2
        self.active = False
        self.hit = False
        self.speed = 1
        self.speed_level = 0
        self.mask = pygame.mask.from_surface(self.image)
        self.energy = Boss.energy
        self.game_lv = 1

    def move(self):
        #由上方生成 移动到屏幕正上
        if self.rect.top < 0:
            self.rect.top += self.speed
        else:
            self.speed = 0            
            self.rect.left += self.speed_level
            
            if self.rect.right >= self.size[0]:
                self.rect.right = self.size[0]
                self.speed_level = -self.speed_level
            
            if self.rect.left <= 0:
                self.rect.left = 0
                self.speed_level = -self.speed_level

    def reset(self):
        self.rect.top, self.rect.left = (-1 *self.rect.height), (self.size[0] - self.rect.width)//2
        self.speed = 1
        
        self.speed_level += 0
        Boss.energy = 2.5 * Boss.energy
        self.energy = Boss.energy
        self.active = True
        
        self.image = pygame.image.load("boss/lv%d.png"%(self.game_lv)).convert_alpha()
        self.image_hit = pygame.image.load("boss/lv%s_hit.png"%(self.game_lv)).convert_alpha()
        self.game_lv += 1

    def _return(self):
        self.active = False
        self.rect.top, self.rect.left = (-1 *self.rect.height), (self.size[0] - self.rect.width)//2
