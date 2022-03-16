from cmath import pi
from genericpath import exists
from select import PIPE_BUF
from xmlrpc.client import Boolean
import pygame
import neat
import random
import os
import time
pygame.font.init()

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 600
FLOOR = WINDOW_HEIGHT - 80

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

FONT = pygame.font.SysFont("American Typewriter", 50)


class Bird:
    IMGS = bird_images
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5
    MAX_VEL = 16
    JUMP_VEL = -10.5

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
        self.anim_count = []

        for i in range(self.ANIMATION_TIME * 4):
            self.anim_count.append(int(i / self.ANIMATION_TIME))
        self.anim_count[:] = [int(1) if x == 3 else x for x in self.anim_count]
        self.anim_count.append(0)

    def jump(self) -> None:
        self.vel = self.JUMP_VEL
        self.tick_count = 0
        self.height = self.y

    def move(self) -> None:
        self.tick_count += 1

        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2

        if d >= self.MAX_VEL:
            d = self.MAX_VEL
        
        if d < 0:
            d -= 2
        
        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    
    def draw(self, win) -> None:
        self.img_count += 1

        self.img = self.IMGS[self.anim_count[self.img_count]]
        self.img_count = self.img_count % (self.ANIMATION_TIME * 4)

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)

        win.blit(rotated_image, new_rect.topleft)
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x, vel) -> None:
        self.x = x
        self.VEL = vel
        self.height = 0

        self.top = 0
        self.bottom = 0
        
        self.pipe_top = pygame.transform.flip(pipe_img, False, True)
        self.pipe_bot = pipe_img

        self.passed = False
        self.set_height()

    def set_height(self) -> None:
        self.height = random.randrange(50, 450)
        self.top = self.height - self.pipe_top.get_height()
        self.bottom = self.height + self.GAP

    def move(self) -> None:
        self.x -= self.VEL
    
    def draw(self, win) -> None:
        win.blit(self.pipe_top, (self.x, self.top))
        win.blit(self.pipe_bot, (self.x, self.bottom))
    
    def collide(self, bird) -> Boolean:
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.pipe_top)
        bot_mask = pygame.mask.from_surface(self.pipe_bot)
 
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bot_offset = (self.x - bird.x, self.bottom - round(bird.y))

        bot_point = bird_mask.overlap(bot_mask, bot_offset)
        top_point = bird_mask.overlap(top_mask, top_offset)

        if bot_point or top_point:
            return True
        return False

class Base:
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y) -> None:
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self) -> None:
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH <= 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH <= 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win) -> None:
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def update_window(win, birds, pipes, base, score) -> None:
    win.fill(0)
    win.blit(bg_img, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = FONT.render("Score:" + str(score), 1, (255, 255, 255))
    win.blit(text, (WINDOW_WIDTH - 10 - text.get_width(), 10))

    for bird in birds:
        bird.draw(win)

    base.draw(win)
    pygame.display.update()

def eval_fitness(genomes, config) -> None:
    nets = []
    gens = []
    birds = []
    
    for _, gen in genomes:
        net = neat.nn.FeedForwardNetwork.create(gen, config)
        nets.append(net)
        birds.append(Bird(220, 350))
        gen.fitness = 0
        gens.append(gen)

    velocity = 5
    base = Base(FLOOR)
    pipes = [Pipe(700, velocity)]
    run = True
    clock = pygame.time.Clock()
    score = 0

    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        
        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipe_top.get_width():
                pipe_index = 1
        else:
            run = False
            break
        
        for n, bird in enumerate(birds):
            bird.move()
            gens[n].fitness += 0.05

            output = nets[n].activate((bird.y, abs(bird.y - pipes[pipe_index].height), abs(bird.y - pipes[pipe_index].bottom)))
            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for n, bird in enumerate(birds):
                if pipe.collide(bird):
                    gens[n].fitness -= 1
                    birds.pop(n)
                    nets.pop(n)
                    gens.pop(n)
              
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
            
            if pipe.x + pipe.pipe_top.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1

            base.VEL += 0.2
            velocity += 0.2
            for pipe in pipes:
                pipe.VEL += 0.2
            
            for bird in birds:
                bird.MAX_VEL += 0.2
            
            for gen in gens:
                gen.fitness += 4
            pipes.append(Pipe(700, velocity))  

        for r in rem:
            pipes.remove(r)  
        
        for n, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= FLOOR or bird.y < -10:
                birds.pop(n)
                nets.pop(n)
                gens.pop(n)

        
        base.move()
        update_window(window, birds, pipes, base, score)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_fitness, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    run(config_path)