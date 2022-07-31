
from collections import deque
import random
import time
import torch
from dqn import Network
# from game import Preprocessing
from game import Game, game_test
import numpy as np
import gym

# Hyperparameters
# BATCH_SIZE = 32
# REPLAY_MEM_SIZE = int(1e6)
AGENT_HISTORY_LEN = 4  # Number of most recent frames given as input to the Q network
TARGET_NET_UPDATE_FREQ = int(1e4)  # C
GAMMA = 0.99  # discount factor used in Q-learning update
ACTION_REPEAT = 4  # Agent only sees every 4th input frame (repeat last action)
UPDATE_FREQ = 4  # Agent selects 4 actions between each pair of successive updates
LEARNING_RATE = 0.25e-3  # learning rate used by RMSProp
GRADIENT_MOMENTUM = 0.95  # RMSProp
SQUARED_GRADIENT_MOMENTUM = 0.95  # RMSProp
MIN_SQUARED_GRADIENT = 0.01  # RMSProp
INITIAL_EXPLORATION = 1  # Initial value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION = 0.1  # final value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION_FRAME = int(1e6)  # num frames epsilon changes linearly
REPLAY_START_SIZE = int(5e4)  # uniform random policy run before learning
NO_OP_MAX = 30  # max num of "do nothing" actions performed by agent at the start of an episode

# DECAY_SLOPE = (1.0-0.1)/(0.0-1e6)
# delta y over delta x
DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
    (0.0-FINAL_EXPLORATION_FRAME)


# optimizer = torch.optim.RMSprop(self.policy_net.parameters())


class Agent:
    def __init__(self, device):

        self.device = device

        # total steps of training, used for epsilon (random action selection)
        self.step = 0

    def select_action(self):
        """ selects action, either random or from model """

        # epsilon = np.interp(self.step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        # epsilon = np.interp(self.step,
        #                     [0, FINAL_EXPLORATION_FRAME],
        #                     [INITIAL_EXPLORATION, FINAL_EXPLORATION])

        # y=mx+c to for step<=FINAL_EXPLORATION_FRAME else epsilon=FINAL_EXPLORATION
        epsilon = DECAY_SLOPE*self.step + \
            1 if self.step <= FINAL_EXPLORATION_FRAME else FINAL_EXPLORATION

        sample = random.random()

        self.step += 1

        if sample < epsilon:
            action = random.randrange(2)
            # return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = "action from Q network..."
                # return self.policy_net(state).max(1)[1].view(1, 1)

        return action
