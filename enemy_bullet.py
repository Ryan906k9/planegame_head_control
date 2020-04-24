import pygame

#尾气
class Bullet(pygame.sprite.Sprite):
    def __init__(self,size):
        
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self.image = pygame.image.load("boss/weiqi.png").convert_alpha()
        self.rect = self.image.get_rect()
        
        self.speed = 10
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        self.rect.top += self.speed
        if self.rect.top > self.size[1]:
            self.active = False
            
    def reset(self,position):   
        self.rect.left, self.rect.top = position
        self.active = True

#boss子弹1
class Bullet1(pygame.sprite.Sprite):
    def __init__(self, size):

        pygame.sprite.Sprite.__init__(self)
        self.size = size        
        self.image = pygame.image.load("boss/boss_b1.png").convert_alpha()
        self.rect_o = self.image.get_rect()
        self.image = pygame.transform.smoothscale(self.image,\
                                                  (int(self.rect_o.width * 0.5),\
                                                   int(self.rect_o.height * 0.5)))
        self.rect = self.image.get_rect()
        #self.rect.left ,self.rect.top = position
        self.speed = 5
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        self.rect.top += self.speed
        if self.rect.top > self.size[1]:
            self.active = False

    def reset(self,position):   
        self.rect.left, self.rect.top = position
        self.active = True

#boss子弹2
class Bullet2(pygame.sprite.Sprite):
    def __init__(self, size):

        pygame.sprite.Sprite.__init__(self)
        self.size = size        
        self.image = pygame.image.load("boss/boss_b2.png").convert_alpha()
        self.rect_o = self.image.get_rect()
        self.image = pygame.transform.smoothscale(self.image,\
                                                  (int(self.rect_o.width * 0.5),\
                                                   int(self.rect_o.height * 0.5)))
        self.rect = self.image.get_rect()
        #self.rect.left, self.rect.top = position
        self.speed = 5
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        self.rect.top += self.speed
        if self.rect.top > self.size[1]:
            self.active = False

    def reset(self,position):   
        self.rect.left, self.rect.top = position
        self.active = True

#boss子弹3
class Bullet3(pygame.sprite.Sprite):
    def __init__(self, size):

        pygame.sprite.Sprite.__init__(self)
        self.size = size        
        self.image = pygame.image.load("boss/boss_b3.png").convert_alpha()
        self.rect_o = self.image.get_rect()
        self.image = pygame.transform.smoothscale(self.image,\
                                                  (int(self.rect_o.width * 0.5),\
                                                   int(self.rect_o.height * 0.5)))
        self.rect = self.image.get_rect()
        #self.rect.left, self.rect.top = position
        self.speed = 5
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        self.rect.top += self.speed
        if self.rect.top > self.size[1]:
            self.active = False

    def reset(self,position):   
        self.rect.left, self.rect.top = position
        self.active = True

#boss子弹a
class Bullet_a(pygame.sprite.Sprite):
    def __init__(self, size):

        pygame.sprite.Sprite.__init__(self)
        self.size = size        
        self.image = pygame.image.load("boss/boss_a1.png").convert_alpha()
        self.rect = self.image.get_rect()
        #self.rect.left, self.rect.top = position
        self.speed = 5
        self.active = False
        self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        self.rect.top += self.speed
        if self.rect.top > self.size[1]:
            self.active = False

    def reset(self,position):   
        self.rect.left, self.rect.top = position
        self.active = True
        

class Lasers1(pygame.sprite.Sprite):
    def __init__(self, size, position):

        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("boss/jiguang4.png").convert_alpha()
        self.rect_o = self.image.get_rect()
        self.size = size
        self.image = pygame.transform.smoothscale(self.image,\
                                                  (int(self.rect_o.width * 0.25),\
                                                   int(self.rect_o.height * 5.9)))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = position
        self.mask = pygame.mask.from_surface(self.image)
        self.speed = 2
        self.active = False
        self.lv = 3

    def move(self):
        self.rect.top += self.speed
        if self.rect.bottom >= self.size[1]:
            self.active = False

    def reset(self,position):
        self.rect.left, self.rect.top = position
        self.active = True
