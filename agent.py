import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import json

from game import BLOCK_SIZE

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(7, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # Move direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        dir_x = (dir_r - dir_l)
        dir_y = (dir_u - dir_d)

        # Calculate Manhattan distance to food
        distance_to_food = abs(game.food.x - head.x) + abs(game.food.y - head.y)

        # Direction to food
        food_left = game.food.x < head.x
        food_right = game.food.x > head.x
        food_up = game.food.y < head.y
        food_down = game.food.y > head.y

        food_x = (food_left - food_right)
        food_y = (food_up - food_down)

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Obstacle proximity indicators
        obstacle_left = game.is_collision(Point(head.x - BLOCK_SIZE, head.y))
        obstacle_right = game.is_collision(Point(head.x + BLOCK_SIZE, head.y))
        obstacle_up = game.is_collision(Point(head.x, head.y - BLOCK_SIZE))
        obstacle_down = game.is_collision(Point(head.x, head.y + BLOCK_SIZE))

        # Body proximity indicators
        body_left = any(Point(head.x - BLOCK_SIZE, head.y) == segment for segment in game.snake[1:])
        body_right = any(Point(head.x + BLOCK_SIZE, head.y) == segment for segment in game.snake[1:])
        body_up = any(Point(head.x, head.y - BLOCK_SIZE) == segment for segment in game.snake[1:])
        body_down = any(Point(head.x, head.y + BLOCK_SIZE) == segment for segment in game.snake[1:])

        danger_straight = ((dir_r and game.is_collision(point_r)) or
                           (dir_l and game.is_collision(point_l)) or
                           (dir_u and game.is_collision(point_u)) or
                           (dir_d and game.is_collision(point_d)))
        danger_right = ((dir_u and game.is_collision(point_r)) or
                        (dir_d and game.is_collision(point_l)) or
                        (dir_l and game.is_collision(point_u)) or
                        (dir_r and game.is_collision(point_d)))
        danger_left = ((dir_d and game.is_collision(point_r)) or
                       (dir_u and game.is_collision(point_l)) or
                       (dir_r and game.is_collision(point_u)) or
                       (dir_l and game.is_collision(point_d)))

        # Snake length
        snake_length = len(game.snake)

        state = [
            # Danger
            danger_straight, danger_right, danger_left,
            # Move direction
            # dir_l, dir_r, dir_u, dir_d,
            dir_x, dir_y,
            # Direction to food
            food_x, food_y,
            # food_left, food_right, food_up, food_down,
            # Calculate Manhattan distance to food
            # distance_to_food,
            # Snake length
            # snake_length,
            # Body proximity indicators
            # body_left, body_right, body_up, body_down,
            # Obstacle proximity indicators
            # obstacle_left, obstacle_right, obstacle_up, obstacle_down,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        # was 200
        if random.randint(0, 150) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        # state0 = torch.tensor(state, dtype=torch.float)
        # prediction = self.model(state0)
        # move = torch.argmax(prediction).item()
        # final_move[move] = 1

        return final_move


def train(model_path=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()

    if model_path is not None:
        agent.model.load(model_path)

    game = SnakeGameAI()

    file_name = input("Enter the name of the model trained on this game: ")

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(file_name)

            # Print to console
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            mean_score = total_score / agent.n_games

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


