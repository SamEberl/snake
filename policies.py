import copy
import random
import time

import numpy as np
from pygame.math import Vector2


def check_fail(snake, cell_number):
    if not 0 <= snake.body[0].x < cell_number or not 0 <= snake.body[0].y < cell_number:
        return True
    for block in snake.body[1:]:
        if block == snake.body[0]:
            return True
    return False


def check_apple(self, snake):
    if self.fruit.pos == snake.body[0]:
        return True


def sim_game(steps, snake, fruit, cell_number):
    # returns whether snake eats itself(0) or the apple(1) and the number of turns to do so
    sim_snake = copy.deepcopy(snake)
    counter = 0
    while 1:
        new_direction = steps.pop(0)
        if new_direction is not None:
            if int(new_direction.dot(sim_snake.cur_direction)) == -1:
                new_direction = sim_snake.cur_direction
            sim_snake.direction = new_direction
        sim_snake.move_snake()
        counter += 1
        if check_fail(sim_snake):
            return False, counter
        elif check_apple(sim_snake):
            return True, counter


def tree_func(snake, fruit_pos, cell_number, cur_direction, prev_path, max_depth, viable_path):
    c_snake = copy.deepcopy(snake)
    if len(prev_path) != 0:
        for i in range(len(prev_path)):
            head = c_snake.body[0] + prev_path[i]
            c_snake.body.insert(0, head)
            c_snake.body.pop()
    # check if snake ate apple
    if len(viable_path) != 0:
        return
    elif fruit_pos == c_snake.body[0]:
        viable_path += prev_path
        return
    # check if step is viable
    elif len(prev_path)+1 <= max_depth:
        left = Vector2(Vector2(0, 1) * cur_direction, Vector2(-1, 0) * cur_direction)
        if check_viable(c_snake, cell_number, left):
            path = copy.deepcopy(prev_path)
            path.append(left)
            tree_func(snake, fruit_pos, cell_number, left, path, max_depth, viable_path)

        straight = cur_direction
        if check_viable(c_snake, cell_number, straight):
            path = copy.deepcopy(prev_path)
            path.append(straight)
            tree_func(snake, fruit_pos, cell_number, straight, path, max_depth, viable_path)

        right = Vector2(Vector2(0, -1) * cur_direction, Vector2(1, 0) * cur_direction)
        if check_viable(c_snake, cell_number, right):
            path = copy.deepcopy(prev_path)
            path.append(right)
            tree_func(snake, fruit_pos, cell_number, right, path, max_depth, viable_path)


def check_viable(c_snake, cell_number, pot_dir):
    if cell_number <= (c_snake.body[0]).x + pot_dir.x or (c_snake.body[0]).x + pot_dir.x < 0:
        return False
    if cell_number <= (c_snake.body[0]).y + pot_dir.y or (c_snake.body[0]).y + pot_dir.y < 0:
        return False
    for block in c_snake.body[1:]:
        if block == (c_snake.body[0] + pot_dir):
            return False
    return True

# def bfs(snake, fruit, cell_number):
#     stack = [[]]
#     while len(stack) != 0:
#         prev_path = stack.pop()
#         cur_direction = prev_path[-1]
#         c_snake = copy.deepcopy(snake)
#         for i in range(len(prev_path)):
#             head = c_snake.body[0] + prev_path[i]
#             c_snake.body.insert(0, head)
#             c_snake.body.pop()
#         left = Vector2(Vector2(0, 1) * cur_direction, Vector2(-1, 0) * cur_direction)
#         straight = cur_direction
#         right = Vector2(Vector2(0, -1) * cur_direction, Vector2(1, 0) * cur_direction)
#         if check_vialble(c_snake, cell_number, left):
#             stack.append(left)
#         if check_vialble(c_snake, cell_number, straight):
#             stack.append(straight)
#         if ...




def dfs(snake, fruit, cell_number):
    viable_path = []
    tree_func(snake, fruit.pos, cell_number, snake.cur_direction, [], 13, viable_path)
    if len(viable_path) == 0:
        pivot_point = Vector2(int(cell_number/2), int(cell_number/2))
        tree_func(snake, pivot_point, cell_number, snake.cur_direction, [], 13, viable_path)
    return viable_path


def iterative_deepening_search(snake, fruit, cell_number):
    # depth first search
    viable_path = []
    for i in range(1, 24):
        tree_func(snake, fruit.pos, cell_number, snake.cur_direction, [], i, viable_path)
        if len(viable_path) != 0:
            print(f'viable_path: {viable_path}')
            return viable_path
    # time.sleep(1)
    return viable_path


def straight_line(snake, fruit, cell_number):
    possible_directions = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]
    snake_to_fruit = fruit.pos - snake.body[0]
    new_direction = Vector2(0, 0)
    for i in copy.deepcopy(possible_directions):
        if int(snake.cur_direction.dot(i)) == -1:
            possible_directions.remove(i)
            continue
        if not 0 <= (snake.body[0]+i).x < cell_number or not 0 <= (snake.body[0]+i).y < cell_number:
            possible_directions.remove(i)
            continue
        for block in snake.body[1:]:
            if block == snake.body[0]+i:
                possible_directions.remove(i)

    for i in possible_directions:
        if int(snake_to_fruit.dot(i)) >= int(snake_to_fruit.dot(new_direction)):
            new_direction = i

    if new_direction == Vector2(0, 0) and len(possible_directions) != 0:
        new_direction = random.choice(possible_directions)

    return [new_direction]



def wave_front(snake, fruit, cell_number):
    possible_directions = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]
    value_grid = np.full((cell_number, cell_number), np.NINF)

    self_crash = False
    counters = [-1]
    queue = [fruit.pos]
    while 1:
        if len(queue) == 0:
            pos_rand_dir = False
            for i in possible_directions:
                pos_rand_dir = True
                if cell_number > snake.body[0].x + i.x >= 0 and cell_number > snake.body[0].y + i.y >= 0:
                    for block in snake.body[1:]:
                        if block == (snake.body[0] + i):
                            pos_rand_dir = False
                    if pos_rand_dir:
                        return [i]
            return [Vector2(0, 0)]  # certain death

        location = queue.pop(0)
        counter = counters.pop(0)
        if cell_number > location.x >= 0 and cell_number > location.y >= 0:
            if value_grid[int(location.x), int(location.y)] < counter:
                if snake.body[0] == location:
                    break
                for block in snake.body[1:]:  # modify length of snake that is checked for
                    if block == location:
                        self_crash = True
                        break
                if self_crash:
                    self_crash = False
                    continue
                value_grid[int(location.x), int(location.y)] = counter
                queue.append(location + possible_directions[0])
                queue.append(location + possible_directions[1])
                queue.append(location + possible_directions[2])
                queue.append(location + possible_directions[3])
                counters += [counter-1, counter-1, counter-1, counter-1]

    # value_grid_trans = value_grid.transpose()
    path = []
    snake_head = snake.body[0]
    cur_value = value_grid[int(snake_head.x), int(snake_head.y)]
    while 1:
        best_dir = Vector2(0, 0)
        for i in possible_directions:
            if cell_number > snake_head.x + i.x >= 0 and cell_number > snake_head.y + i.y >= 0:
                value = value_grid[int(snake_head.x + i.x), int(snake_head.y + i.y)]
                if value > cur_value:
                    best_dir = i
                    cur_value = value
        if cur_value == value_grid[int(snake_head.x), int(snake_head.y)]:
            break
        else:
            path.append(best_dir)
            snake_head = snake_head + best_dir

    return path


def hamiltonian_basic(snake, fruit, cell_number):
    if snake.body[0] != Vector2(0, 0):
        if snake.body[0].y - 1 >= 0:
            return [Vector2(0, -1)]
        elif snake.body[0].x - 1 >= 0:
            return [Vector2(-1, 0)]

    path = [Vector2(0, 1)] * (cell_number-1)
    path += [Vector2(1, 0)]

    for i in range(cell_number//2):
        path += [Vector2(1, 0)] * (cell_number-2)
        path += [Vector2(0, -1)]
        path += [Vector2(-1, 0)] * (cell_number-2)
        if i != (cell_number//2 - 1):
            path += [Vector2(0, -1)]
        else:
            path += [Vector2(-1, 0)]

    return path


