import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import os
from collections import deque

import numpy as np
from pygame.math import Vector2


class Agent:
    def __init__(self, trained):
        self.possible_directions = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]
        # self.memory = deque(maxlen=20000)  # takes in tuples of state_0, action, reward, state_1, game_over
        self.memory = deque(maxlen=2_000)  # takes in state_0
        self.batch_size = 300
        # self.q_net = DQN()
        self.q_net = LinearNet()
        self.target_net = copy.deepcopy(self.q_net)
        self.gamma = 0.9
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.iteration = 0
        self.max_iterations = 10_000
        if trained:
            self.q_net.load_state_dict(torch.load('./model/lin_model.pth'))
            self.q_net.eval()

    def get_state(self, snake, fruit, cell_number):
        state = [
            # Danger up
            self.check_fail_after_action(snake, cell_number, self.possible_directions[0]),
            # Apple up
            self.check_apple_after_action(snake, fruit, self.possible_directions[0]),

            # Danger right
            self.check_fail_after_action(snake, cell_number, self.possible_directions[1]),
            # Apple right
            self.check_apple_after_action(snake, fruit, self.possible_directions[1]),

            # Danger down
            self.check_fail_after_action(snake, cell_number, self.possible_directions[2]),
            # Apple down
            self.check_apple_after_action(snake, fruit, self.possible_directions[2]),

            # Danger left
            self.check_fail_after_action(snake, cell_number, self.possible_directions[3]),
            # Apple left
            self.check_apple_after_action(snake, fruit, self.possible_directions[3]),

            # Move direction
            snake.cur_direction == self.possible_directions[0],  # up
            snake.cur_direction == self.possible_directions[1],  # right
            snake.cur_direction == self.possible_directions[2],  # down
            snake.cur_direction == self.possible_directions[3],  # left

            # Food location
            fruit.pos.y < snake.body[0].y,  # food up
            fruit.pos.x > snake.body[0].x,  # food right
            fruit.pos.y > snake.body[0].y,  # food down
            fruit.pos.x < snake.body[0].x,  # food left
        ]
        return torch.tensor(state, dtype=torch.float32)

    def train_step(self, state_0, snake, fruit, cell_number):
        q_0 = self.q_net(state_0)
        if 100-self.iteration > random.randint(0, 100):
            action = self.possible_directions[random.randint(0, 3)]
        else:
            action = self.possible_directions[torch.argmax(q_0)]
        reward, state_1, game_over = self.take_step(action, snake, fruit, cell_number)
        print(reward)
        if game_over:
            q_1 = torch.tensor(-10, dtype=torch.float32)
        else:
            q_1 = reward + self.gamma * torch.max(self.target_net.forward(state_1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_0, q_1)
        loss.backward()
        self.optimizer.step()
        self.iteration += 1

        if self.iteration % 1000 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{self.iteration:>5d}/{self.max_iterations:>5d}]")

        return action

    def train_dqn(self, snake, fruit, cell_number):
        # state_0 = self.construct_state(snake, fruit, cell_number)
        state_0 = self.get_state(snake, fruit, cell_number)
        self.memory.append((state_0, snake, fruit, cell_number))
        action = self.train_step(state_0, snake, fruit, cell_number)

        if self.iteration % 1000 == 0:
            self.target_net = copy.deepcopy(self.q_net)
            if len(self.memory) > self.batch_size:
                mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
            else:
                mini_sample = self.memory
            # states_0, snakes, fruits, cell_numbers = zip(*mini_sample)
            for states_0, snakes, fruits, cell_numbers in mini_sample:
                self.train_step(states_0, snakes, fruits, cell_numbers)

        if self.iteration > self.max_iterations:
            self.q_net.save()
            exit()

        return [action]

    def play_dqn(self, snake, fruit, cell_number):
        # state = self.construct_state(snake, fruit, cell_number)
        state = self.get_state(snake, fruit, cell_number)
        action = self.possible_directions[torch.argmax(self.q_net(state))]
        return [action]

    def construct_state(self, snake, fruit, cell_number):
        # channels are head, body, fruit
        state = torch.zeros((1, 3, cell_number, cell_number))
        state[0, 0, int(snake.body[0].x), int(snake.body[0].y)] = 1
        for block in snake.body[1:]:
            state[0, 1, int(block.x), int(block.y)] = 1
        state[0, 2, int(fruit.pos.x), int(fruit.pos.y)] = 1
        return state


    def take_step(self, action, snake, fruit, cell_number):
        sim_snake = copy.deepcopy(snake)
        if action is not None:
            if int(action.dot(sim_snake.cur_direction)) == -1:
                action = sim_snake.cur_direction
            sim_snake.direction = action
        sim_snake.move_snake()
        reward = 0
        game_over = False
        if self.check_fail(sim_snake, cell_number):
            reward -= 100
            game_over = True
        elif self.check_apple(sim_snake, fruit):
            reward += 100
        # elif self.distance_to_apple_shorter(snake, sim_snake, fruit):
        #     reward += 1


        if game_over:
            state_1 = torch.zeros((1, 3, cell_number, cell_number))
        else:
            # state_1 = self.construct_state(sim_snake, fruit, cell_number)
            state_1 = self.get_state(sim_snake, fruit, cell_number)

        return reward, state_1, game_over

    def distance_to_apple_shorter(self, snake, sim_snake, fruit):
        dist_old = snake.body[0] - fruit.pos
        dist_new = sim_snake.body[0] - fruit.pos

        if (dist_old.x**2 + dist_old.y**2) < (dist_new.x**2 + dist_new.y**2):
            return True
        else:
            return False

    def check_fail(self, snake, cell_number):
        if not 0 <= snake.body[0].x < cell_number or not 0 <= snake.body[0].y < cell_number:
            return True
        for block in snake.body[1:]:
            if block == snake.body[0]:
                return True

    def check_apple(self, snake, fruit):
        if fruit.pos == snake.body[0]:
            return True

    def check_fail_after_action(self, snake, cell_number, action):
        if not 0 <= snake.body[0].x+action.x < cell_number or not 0 <= snake.body[0].y+action.y < cell_number:
            return True
        for block in snake.body[:-1]:
            if block == snake.body[0]+action:
                return True
        return False

    def check_apple_after_action(self, snake, fruit, action):
        if fruit.pos == snake.body[0]+action:
            return True
        else:
            return False


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)  # in: 1x3x20x20
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1) # in: 1x4x10x10
        self.fc1 = nn.Linear(200, 16)  # in: 1x8x5x5
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 200)
        x = F.relu(self.fc1(x))
        x = torch.squeeze(self.fc2(x))
        return x

    def save(self, file_name='cnn_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()

        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, file_name='lin_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


# class Network:
#     def __init__(self):
#         self.cell_number = 10
#         self.fc1 = {'weights': np.random.randint(0, 2, (16, (self.cell_number ** 2)*2)), 'bias': 0}
#         self.fc2 = {'weights': np.random.randint(0, 2, (4, 16)), 'bias': 0}
#
#     def forward(self, snake, fruit, cell_number):
#         x = np.zeros((self.cell_number ** 2)*2)
#         for i in snake.body:
#             x[int(i.x + cell_number*i.y)] = 1
#         x[int(fruit.pos[0] + fruit.pos[1]*cell_number + self.cell_number**2)] = 1
#         x = self.fc1['weights'] @ x + self.fc1['bias']
#         x = x * (x > 0)
#         x = self.fc2['weights'] @ x + self.fc2['bias']
#         x = np.exp(x)
#         x = x / (np.sum(x) + 0.0000001)
#         return np.argmax(x)
#
#     def backward(self, x):
#         s = 1 / (1+np.exp(-x))
#         s_diag = np.diag(s)
#         s_back = np.identity() @ s_diag - s_diag @ s_diag.T
#
#     def inference(self, snake, fruit, cell_number):
#         dirs = [Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
#         i = self.forward(snake, fruit, cell_number)
#         print(i)
#         return [dirs[i]]


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# model = LinearNet().to(device)
# print(model)
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    directions = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        epsilon = 1
        if random.random() < epsilon:
            return random.choice(directions)
        else:
            pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")