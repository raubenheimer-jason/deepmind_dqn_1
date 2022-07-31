

from collections import deque
import random
import time
import torch
from dqn import Network
# from game import Preprocessing
from game import Game, Preprocessing, game_test
import numpy as np
import gym
from itertools import count

from agent import Agent

REPLAY_MEM_SIZE = int(1e6)
BATCH_SIZE = 32


def main():
    # start = time.time()
    # game_test()

    # print(time.time()-start)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # return

    # game holds the env
    game = Game()

    agent = Agent()

    # preprocess = Preprocessing()

    # * Initialize replay memory D to capacity N
    replay_mem = deque(maxlen=REPLAY_MEM_SIZE)  # replay_mem is D
    # Need to fill the replay_mem (to REPLAY_START_SIZE) with the results from random actions
    #   -> maybe do this in the main loop and just select random until len(replay_mem) >= REPLAY_START_SIZE

    # get number of actions
    # https://stackoverflow.com/questions/63113154/how-to-check-out-actions-available-in-openai-gym-environment
    # num_actions = env.action_space.n
    # num_actions = game.num_actions

    # * Initialize action-value function Q with random weights Theta
    # initialise policy_net
    policy_net = Network().to(device)

    # * Initialize target action-value function Q_hat with weights Theta_bar = Theta
    # initialise target_net
    target_net = Network().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # * For episode = 1, M do
    for episode in count():

        # * Initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)
        # preprocessed current state (from initial reset)
        s_t = [game.current_obs]  # s_t=1 initialise with x_1
        phi_t = preprocess.process(s_t)  # phi_t=1, preprocessed sequence

        # * For t = 1, T do
        for t in count():

            # * With probability epsilon select a random action a_t
            # * otherwise select a_t = argmax_a Q(phi(s_t),a;Theta)
            a_t = agent.select_action()

            # * Execute action a_t in emulator and observe reward r_t and image x_t+1
            r_t, x_tplus1 = game.step(a_t)

            # * Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)
            s_tplus1 = [s_t, a_t, x_tplus1]
            # s.append(a_t)
            # s.append(x_tplus1)  # creating s_t+1
            # calculate preprocessed phi at t+1 using s_t+1
            phi_tplus1 = preprocess.process(s_tplus1)  # phi_t+1

            # * Store transition (phi_t, a_t, r_t, phi_t+1) in D
            transition = (phi_t, a_t, r_t, phi_tplus1)
            replay_mem.append(transition)  # replay_mem is D

            # * Sample random minibatch of transitions (phi_j, a_j, r_j, phi_j+1) from D
            minibatch = random.sample(replay_mem, BATCH_SIZE)

            # * Set y_j = r_j if episode terminates at step j+1
            # * otherwise set y_j = r_j + gamma * max_a_prime Q_hat(phi_j+1, a_prime; Theta_bar)

            # * Perform a gradient descent step on (y_j - Q(phi_j, a_j; Theta))^2 with respect to the network parameters Theta

            # * Every C steps reset Q_hat = Q

    # * End For
    # * End For


if __name__ == "__main__":
    # print("Agent")
    # init_game()

    main()
