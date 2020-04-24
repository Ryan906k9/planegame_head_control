import pygame

class Shield(pygame.sprite.Sprite):
    energy = 1500
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.image1 = pygame.image.load("images/shield01.png").convert_alpha()
        self.image2 = pygame.image.load("images/shield02.png").convert_alpha()
        self.mask = pygame.mask.from_surface(self.image1)
        self.rect = self.image1.get_rect()
        self.active = False
        self.hit = False


    def move(self,position):
        self.rect.left, self.rect.top = position

    def reset(self):
        self.active = True
        Shield.energy *=1.2
        self.energy = Shield.energy
