from statistics import mean
import torch
import random
import numpy as np
from collections import deque
from snake_game import IA_SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from ploter import plot


# parametre deque memory

# could save 100_000 items into memory
MAX_MEMORY = 100_000
# The batch size defines the number of samples that will be propagated through the network.
BATCH_SIZE = 1000


learning_rate = 0.001


# class Agent
# GAME (snake)
# MODEL (pyTorch) : Linear_QNet : model.predict(state) -> action
#  training:
#   state : get_state(game)
#   action : get_move(state)
#     - model.predict() (pyTorch)
#   reward, game_over, score = game.play_step(action)
#   new_state
#   remember
#   model.train

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # parameter which control the randomness
        self.gamma = 0.8  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # call popleft() function
        self.model = Linear_QNet(11, 256, 3) #TODO
        self.trainer = QTrainer(self.model, learning_rate, self.gamma) 
       

    # we need a function to get_state for the training
    def get_state(self, game):
        # the state is calculated with the 11 input variables
        # [ danger straight, danger right, danger left, [ 0 , 0, 0,
        #   direction left, direction right,              0, 0, 0, 0
        #   direction up, direction  down,                0, 0, 0, 0]
        #   food left, food right,
        #   food up, food down
        #   ]

        # snake head info
        head = game.snake[0]
        # we create point around the head to pass it in parameter to detect danger zone
        point_l = Point(head.x - 20, head.y)  # 20 is block size in the game
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)  # graph 0,0 en haut a gauche
        point_d = Point(head.x, head.y + 20)

        # direction info
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State info

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_l)),
            # danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # direction move
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,  # food left
            game.head.x < game.food.x,  # food right
            game.food.y < game.head.y,  # food up
            game.head.y < game.food.y  # food down
        ]

        return np.array(state, dtype=int)

    # we need a function to remember the state, action, reward, and next_state of the snake
    def remember(self, state, action, reward, next_state, game_over_state):
        # popleft if the max memory is througt
        self.memory.append((state, action, reward, next_state, game_over_state))

    # we need a function to train the long memory
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else :
            mini_sample = self.memory

        states, actions, rewards, next_states, game_over_states = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_over_states)
        # sans la fonction zip() :
        # for state,action, reward, next_state, game_over_state in mini_sample:
        #  self.trainer.train_step(state, action, reward, next_state, game_over_state)

    # we need a function to train the short memory in one step
    def train_short_memory(self, state, action, reward, next_state, game_over_state):
        self.trainer.train_step(state, action, reward, next_state, game_over_state)

    # we need a function to get the action of the snake based the state
    def get_action(self, state):
        # random move : tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        # more games we have less random move there are
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) # 0<= move <= 1 et move <= 2
            final_move[move] = 1
         
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            # call the forward fonction into model
            prediction = self.model(state0)
            # on recupère une unique valeure maximale de la prédiction
            move = torch.argmax(prediction).item()
            final_move[move] = 1
         
            
        return final_move

# definition of the function to the training
def train():
    # training:
    #   state : get_state(game)
    #   action : get_move(state)
    #     - model.predict() (pyTorch)
    #   reward, game_over, score = game.play_step(action)
    #   new_state
    #   remember
    #   model.train

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = IA_SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move based on current state
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over_state, score = game.play_step(final_move)

        state_new = agent.get_state(game)

        # train the short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over_state)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over_state)

        if game_over_state:
            # train the long memory : experience replay  && plot the result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean = total_score / agent.n_games
            plot_mean_scores.append(mean)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()
