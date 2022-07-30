
from collections import deque
import torch

# from game import init_game
from dqn import Network

# from game import preprocessing_t
from game import Preprocessing

import numpy as np

# Hyperparameters
BATCH_SIZE = 32
REPLAY_MEM_SIZE = int(1e6)
AGENT_HISTORY_LEN = 4  # Number of most recent frames given as input to the Q network
TARGET_NET_UPDATE_FREQ = int(1e4)  # C
GAMMA = 0.99  # discount factor used in Q-learning update
ACTION_REPEAT = 4  # Agent only sees every 4th input frame
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


def game():
    import gym
    import matplotlib.pyplot as plt
    from PIL import Image

    # env = gym.make("CartPole-v1", render_mode="human", new_step_api=True)
    # obs = env.reset()
    # for _ in range(10):
    #     # env.render()
    #     env.step(env.action_space.sample())  # take a random action

    # return

    # env = gym.make("Breakout-v4", new_step_api=True)
    # env = gym.make("ALE/Breakout-v5",
    #                render_mode='rgb_array',
    #                new_step_api=True)
    env = gym.make("ALE/Breakout-v5",
                   render_mode="rgb_array",  # or human
                   new_step_api=True)
    # env.metadata["render_fps"] = 60
    print(env.unwrapped.get_action_meanings())
    print(env.action_space.n)

    observation, info = env.reset(seed=42, return_info=True)
    print(f"observation.shape: {observation.shape}")

    p = Preprocessing()

    for _ in range(1):
        action = env.action_space.sample()  # random action

        # observation.shape = (210,160,3)
        observation, reward, terminated, truncated, info = env.step(action)

        # preprocessing_t(observation)

        obs_t = p.process(observation)

        print(obs_t.dtype)
        print(type(obs_t))
        print(obs_t.shape)

        break

        print(f"before: {observation.shape}")
        plt.imshow(observation, cmap='hot', interpolation='nearest')
        plt.show()
        # g_obs = rgb2gray(observation)
        proc = preprocessing(observation)
        print(f"after: {proc.shape}")

        print(proc[:, :, 0])

        # print(g_obs[80, 80])
        # plt.imshow(proc, cmap='hot', interpolation='nearest')
        plt.matshow(proc)
        # plt.imshow(proc, cmap='hot', interpolation='nearest')
        plt.show()

        if terminated or truncated:
            # print(f"resetting {terminated}  {truncated}")
            observation, info = env.reset(return_info=True)

        print("------------------------------------")

    env.close()

    # import gym
    # env = gym.make("CartPole-v1",
    #                render_mode='rgb_array',
    #                new_step_api=True)
    # observation, info = env.reset(seed=42, return_info=True)

    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)

    #     if terminated:
    #         observation, info = env.reset(return_info=True)
    # env.close()

    # # print(env.render(mode='rgb_array'))
    # for _ in range(1):
    #     random_action = env.action_space.sample()
    #     # print(random_action)
    #     obs, reward, terminated, truncated, info = env.step(random_action)
    #     # print()
    # # print(env.render())


def main():
    game()

    return

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # * Initialize replay memory D to capacity N
    replay_mem = deque(maxlen=REPLAY_MEM_SIZE)
    # Need to fill the replay_mem (to REPLAY_START_SIZE) with the results from random actions
    #   -> maybe do this in the main loop and just select random until len(replay_mem) >= REPLAY_START_SIZE

    # get number of actions
    # https://stackoverflow.com/questions/63113154/how-to-check-out-actions-available-in-openai-gym-environment
    num_actions = env.action_space.n

    # * Initialize action-value function Q with random weights Theta
    # initialise policy_net
    policy_net = Network().to(device)

    # * Initialize target action-value function Q_hat with weights Theta_bar = Theta
    # initialise target_net
    target_net = Network().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # * For episode = 1, M do
    # *     Initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)

    # *    For t = 1, T do
    # *         With probability epsilon select a random action a_t
    # *         otherwise select a_t = argmax_a Q(phi(s_t),a;Theta)

    # *         Execute action a_t in emulator and observe reward r_t and image x_t+1

    # *         Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)

    # *         Store transition (phi_t, a_t, r_t, phi_t+1) in D

    # *         Sample random minibatch of transitions (phi_j, a_j, r_j, phi_j+1) from D

    # *         Set y_j = r_j if episode terminates at step j+1
    # *         otherwise set y_j = r_j + gamma * max_a_prime Q_hat(phi_j+1, a_prime; Theta_bar)

    # *         Perform a gradient descent step on (y_j - Q(phi_j, a_j; Theta))^2 with respect to the network parameters Theta

    # *         Every C steps reset Q_hat = Q
    # *    End For
    # * End For


if __name__ == "__main__":
    # print("Agent")
    # init_game()

    main()
