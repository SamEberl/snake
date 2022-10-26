import random
import pygame
# import torch
import sys
from pygame.math import Vector2
import policies
import nn


class SNAKE:
    def __init__(self, body=None):
        if body is None:
            self.body = [Vector2(2, 2), Vector2(1, 2), Vector2(0, 2)]
            # self.body = [Vector2(6, 6), Vector2(6, 5), Vector2(6, 4)]
        else:
            self.body = body
        self.direction = Vector2(0, 0)
        self.cur_direction = Vector2(1, 0)
        self.new_block = False
        self.steps = 0

    def move_snake(self):
        self.steps += 1
        if self.direction.x == 0 and self.direction.y == 0:
            self.steps = 0
        elif self.new_block:
            head = self.body[0] + self.direction
            self.body.insert(0, head)
            self.cur_direction = self.direction
            self.new_block = False
        else:
            head = self.body[0] + self.direction
            self.body.insert(0, head)
            self.body.pop()
            self.cur_direction = self.direction

    def add_block(self):
        self.new_block = True


class FRUIT:
    def __init__(self, snake_body, cell_number):
        self.pos = Vector2(-1, -1)
        self.cell_number = cell_number
        self.randomize(snake_body)
        self.pos = Vector2(4, 2)

    def randomize(self, snake_body):
        free_locations = []
        for i in range(self.cell_number):
            for j in range(self.cell_number):
                if Vector2(i, j) not in snake_body:
                    free_locations.append(Vector2(i, j))
                else:
                    continue
        if len(free_locations) != 0:
            self.pos = random.choice(free_locations)
        else:
            print('You Win')


class MAIN:
    def __init__(self, frames, cell_size=40, cell_number=20):
        self.snake = SNAKE()
        self.fruit = FRUIT(self.snake.body, cell_number)
        self.cell_size = cell_size
        self.cell_number = cell_number
        self.frames_per_sec = frames
        self.new_direction = None
        self.new_direction_list = []

    def update(self):
        self.snake.move_snake()
        self.check_fail(self.snake)
        self.check_apple(self.snake)

    def check_fail(self, snake):
        if not 0 <= snake.body[0].x < self.cell_number or not 0 <= snake.body[0].y < self.cell_number:
            self.game_over()
        for block in snake.body[1:]:
            if block == snake.body[0]:
                self.game_over()

    def check_apple(self, snake):
        if self.fruit.pos == snake.body[0]:
            self.fruit.randomize(self.snake.body)
            self.snake.add_block()
            # time.sleep(5)

    def game_over(self):
        # print(f'snake died after {self.snake.steps} steps on the tile {self.snake.body[0]}')
        # print(f'Score: {str(len(self.snake.body) - 3)}')
        # print('--------------------------')
        self.snake.steps = 0
        self.snake = SNAKE()
        self.fruit = FRUIT(self.snake.body, self.cell_number)
        self.new_direction = None
        self.new_direction_list = []

    def draw_elements(self, fruit_pos, screen, game_font):
        # fruit
        pygame.draw.circle(screen, pygame.Color(150, 0, 0),
                           (fruit_pos.x * self.cell_size + self.cell_size / 2,
                            fruit_pos.y * self.cell_size + self.cell_size / 2), self.cell_size / 2)
        # snake
        for i, block in enumerate(self.snake.body, start=0):
            x_pos = int(block.x * self.cell_size)
            y_pos = int(block.y * self.cell_size)
            block_rect = pygame.Rect(x_pos, y_pos, self.cell_size, self.cell_size)
            if block == self.snake.body[0]:
                pygame.draw.rect(screen, (0, 50, 150), block_rect, 20)
            else:
                gradient = (-1/len(self.snake.body))*i+1
                pygame.draw.rect(screen, (0, 150*(1-gradient), 150*gradient), block_rect, 7)

        # score
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, (200, 200, 200))
        score_x = int(self.cell_size * self.cell_number - 50)
        score_y = int(self.cell_size * self.cell_number - 50)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        screen.blit(score_surface, score_rect)

    def play_game(self, policy=None, render_display=True):
        # expects a policy function that returns a list of Vector2 telling the snake how to move.
        if render_display:
            screen = pygame.display.set_mode((self.cell_number * self.cell_size, self.cell_number * self.cell_size))
            clock = pygame.time.Clock()
            game_font = pygame.font.Font(None, 25)

            screen.fill((0, 0, 0))
            self.draw_elements(self.fruit.pos, screen, game_font)
            pygame.display.update()

        while 1:
            self.new_direction = self.play_human()
            if policy is not None:
                if len(self.new_direction_list) == 0:
                    self.new_direction_list = policy(self.snake, self.fruit, self.cell_number)
                if len(self.new_direction_list) != 0:
                    self.new_direction = self.new_direction_list.pop(0)
                else:
                    self.new_direction = Vector2(0, 0)
                    print('didn\'t find a suitable path with current policy')

            if self.new_direction is not None:
                # print(f'new_dir: {new_direction}')
                # print(f'new_dir_list: {new_direction_list}')
                if int(self.new_direction.dot(self.snake.cur_direction)) == -1:
                    self.new_direction = self.snake.cur_direction
                self.snake.direction = self.new_direction
            self.update()

            if render_display:
                screen.fill((0, 0, 0))
                self.draw_elements(self.fruit.pos, screen, game_font)
                pygame.display.update()
                clock.tick(self.frames_per_sec)

    def play_human(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return Vector2(0, -1)
                if event.key == pygame.K_RIGHT:
                    return Vector2(1, 0)
                if event.key == pygame.K_DOWN:
                    return Vector2(0, 1)
                if event.key == pygame.K_LEFT:
                    return Vector2(-1, 0)
                if event.key == pygame.K_r:
                    self.snake.steps = 0
                    self.snake = SNAKE()
                    self.fruit = FRUIT(self.snake.body, self.cell_number)




pygame.init()
game = MAIN(frames=20, cell_size=40, cell_number=20)


# agent = nn.Agent(trained=False)

# game.play_game()
# game.play_game(policies.straight_line)
# game.play_game(policies.dfs)
# game.play_game(policies.iterative_deepening_search)
game.play_game(policies.wave_front)
# game.play_game(policies.hamiltonian_basic)
# game.play_game(agent.train_dqn, render_display=False)
# game.play_game(agent.play_dqn)

