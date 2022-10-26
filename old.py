import random
import time
import sys
import pygame
from enum import Enum


class Direction(Enum):
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4

window_width = 1000
window_height = 1000
field_width = 50
field_height = 50

cell_size = 40
cell_number = 20

pygame.init()
pygame.display.set_caption("Snake")
screen = pygame.display.set_mode((cell_size*cell_number, cell_size*cell_number))

clock = pygame.time.Clock()
framerate = 10

snake_position = [25, 25]
snake_body = [[25, 25],
              [24, 25],
              [23, 25]]

food_position = [30, 30]

scale = 40
border = 1

global score
score = 0

def handle_keys(direction):
    new_direction = direction
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event == pygame.K_UP and direction != Direction.DOWN:
            new_direction = Direction.UP
        if event == pygame.K_DOWN and direction != Direction.UP:
            new_direction = Direction.DOWN
        if event == pygame.K_RIGHT and direction != Direction.LEFT:
            new_direction = Direction.RIGHT
        if event == pygame.K_LEFT and direction != Direction.RIGHT:
            new_direction = Direction.LEFT
    return new_direction

def move_snake(direction):
    if direction == Direction.UP:
        snake_position[1] -= 1
    if direction == Direction.DOWN:
        snake_position[1] += 1
    if direction == Direction.RIGHT:
        snake_position[0] += 1
    if direction == Direction.LEFT:
        snake_position[0] -= 1
    snake_body.insert(0, list(snake_position))

def generate_new_food():
    food_position[0] = random.randint(0, 50)
    food_position[1] = random.randint(0, 50)

def get_food():
    global score
    if snake_position[0] == food_position[0] and snake_position[1] == food_position[1]:
        score += 1
        generate_new_food()
    else:
        snake_body.pop()

def repaint():
    window.fill(pygame.Color(0, 0, 0))
    # pygame.draw.rect(window, pygame.Color(250, 50, 50), pygame.Rect(0, 0, scale, scale))
    # pygame.draw.rect(window, pygame.Color(250, 50, 50), pygame.Rect(0, (field_height - 1) * scale, scale, scale))
    # pygame.draw.rect(window, pygame.Color(250, 50, 50), pygame.Rect((field_height - 1) * scale, 0, scale, scale))
    # pygame.draw.rect(window, pygame.Color(250, 50, 50), pygame.Rect((field_height - 1) * scale, (field_height - 1) * scale, scale, scale))
    for body in snake_body:
        pygame.draw.rect(window, pygame.Color(0, 255, 0), pygame.Rect((body[0]-1/2)*scale, (body[1]-1/2)*scale, scale, scale))
    for block in range(cell_number):
        pygame.draw.rect(window, pygame.Color(50, 50, 50), pygame.Rect((border-1/2)*scale, block*scale, scale, scale))
        pygame.draw.rect(window, pygame.Color(50, 50, 50), pygame.Rect((cell_number-border-1/2)*scale, block * scale, scale, scale))
    for block in range(cell_number):
        pygame.draw.rect(window, pygame.Color(50, 50, 50), pygame.Rect(block*scale, (border-1/2)*scale, scale, scale))
        pygame.draw.rect(window, pygame.Color(50, 50, 50), pygame.Rect(block * scale, (cell_number-border-1/2)*scale, scale, scale))
    pygame.draw.circle(window, pygame.Color(255, 0, 0), (food_position[0]*scale, food_position[1]*scale), scale/2)
    print("body: ", snake_position)
    print("food: ", food_position)

def game_over_message():
    font = pygame.font.SysFont('Arial', scale*3)
    render = font.render(f"Score: {score}", True, pygame.Color(255,255,255))
    rect = render.get_rect()
    rect.midtop = (cell_size*cell_number/2, cell_size*cell_number/2)
    window.blit(render, rect)
    pygame.display.flip()
    time.sleep(55)
    pygame.quit()
    exit(0)

def game_over():
    if snake_position[0] < border or snake_position[0] > field_width-border:
        game_over_message()
    if snake_position[1] < border or snake_position[1] > field_height-border:
        game_over_message()
    for blob in snake_body[1:]:
        if snake_position[0] == blob[0] and snake_position[1] == blob[1]:
            game_over_message()

def paint_hud():
    font = pygame.font.SysFont('Arial', scale*2)
    render = font.render(f"Score: {score}", True, pygame.Color(255,255,255))
    rect = render.get_rect()
    window.blit(render, rect)
    pygame.display.flip()

def game_loop():
    direction = Direction.RIGHT
    while True:
        direction = handle_keys(direction)
        move_snake(direction)
        get_food()
        repaint()
        game_over()
        paint_hud()
        pygame.display.update()
        clock.tick(framerate)

if __name__ == "__main__":
    game_loop()

