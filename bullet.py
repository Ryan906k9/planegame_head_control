import pygame

class Bullet1(pygame.sprite.Sprite):
    def __init__(self,position):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("images/bullet1.png").convert_alpha()
        self.rect = self.image.get_rect()
        #self.rect.left, self.rect.top = position
        self.speed = 11
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        if self.rect.top > 0:
            self.rect.top -= self.speed
        else:
            self.active = False

    def reset(self,position):
        self.rect.left, self.rect.top = position
        self.active = True
       
class Bullet2(pygame.sprite.Sprite):
    def __init__(self,position):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("images/bullet2.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        self.speed = 12
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        if self.rect.top > 0:
            self.rect.top -= self.speed
        else:
            self.active = False

    def reset(self,position):
        self.rect.left, self.rect.top = position
        self.active = True
       
class Bullet3(pygame.sprite.Sprite):
    def __init__(self,position):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("images/bullet3.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        self.speed = 13
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        if self.rect.top > 0:
            self.rect.top -= self.speed
        else:
            self.active = False

    def reset(self,position):
        self.rect.left, self.rect.top = position
        self.active = True
        
class Bullet4(pygame.sprite.Sprite):
    def __init__(self,position):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("images/bullet5.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        self.speed = 10
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        if self.rect.top > 0:
            self.rect.top -= self.speed
        else:
            self.active = False

    def reset(self,position):
        self.rect.left, self.rect.top = position
        self.active = True

#飞弹
class Bullet5(pygame.sprite.Sprite):
    def __init__(self,position):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("images/feidan.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        #初速度
        self.speed = 3
        #加速度
        self.a = -1
        #时间控制
        self.delay = 100
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        if not self.delay % 10:
            self.speed = self.speed + self.a
        
        if not self.delay:
            self.delay = 100
        if self.rect.top > 0:
            self.rect.top += self.speed
        else:
            self.active = False

        self.delay -= 1

    def reset(self,position):
        self.speed = 3
        self.rect.left, self.rect.top = position
        self.active = True
        

class My_lasers(pygame.sprite.Sprite):
    def __init__(self):

        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("boss/sweep.png").convert_alpha()
        self.rect_o = self.image.get_rect()
        self.image = pygame.transform.smoothscale(self.image,\
                                                  (int(self.rect_o.width * 4.6),\
                                                   int(self.rect_o.height * 0.125)))
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.speed = 4
        self.a = 0
        #加速度
        self.active = False

    def move(self):
        self.speed += self.a
        self.rect.top -= self.speed
        if self.rect.top <= 0:
            self.active = False

    def reset(self,position):
        self.rect.left, self.rect.top = position
        self.active = True

