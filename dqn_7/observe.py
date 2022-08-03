# used to see the model play the game

from main import Network, State, select_action, AGENT_HISTORY_LEN, ACTION_REPEAT
import gym
import torch
import itertools
import time
from random import randrange


MAX_INITIAL_RANDOM_ACTIONS = 30


def observe():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    env = gym.make("ALE/Breakout-v5",
                   render_mode="human",
                   new_step_api=True)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    # env = gym.wrappers.FrameStack(env, 4, new_step_api=True)
    #! need to add "max pix value" to observation...
    num_actions = env.action_space.n
    # env_obs_space = env.observation_space

    net = Network(num_actions, AGENT_HISTORY_LEN).to(device)

    state = State()

    # net.load("../models/2022-08-02__11-52-56/2900k.pkl", device)
    # net.load("../models/2022-08-03__08-52-14/100k.pkl", device)
    net.load("../models/2022-08-03__11-00-27/100k.pkl", device)
    print("done loading")

    obs = env.reset()
    state.add_frame(obs)

    # beginning_episode = True

    num_rand_actions = randrange(30)
    # num_rand_actions = 30*ACTION_REPEAT
    action = randrange(num_actions)

    episode_step = 0

    for _ in itertools.count():
        # step=-1 so we never select a random action

        if episode_step % ACTION_REPEAT == 0:
            if episode_step < num_rand_actions:
                # random number of random actions at the start of each episode
                action = env.action_space.sample()
                print(f"{episode_step}\trandom action: {action}")
            else:
                phi_t = state.get_state()
                action = select_action(num_actions, -1, phi_t, net, device)
                print(f"{episode_step}\tselected action: {action}")

        # if beginning_episode:
        #     action = 1  # "FIRE"
        #     beginning_episode = False

        obs, _, term, trun, _ = env.step(action)
        state.add_frame(obs)
        time.sleep(0.02)

        if term or trun:
            obs = env.reset()
            state.add_frame(obs)
            # beginning_episode = True
            num_rand_actions = randrange(30)
            # num_rand_actions = 30*ACTION_REPEAT
            print(f"num_rand_actions: {num_rand_actions}")
            episode_step = 0

        episode_step += 1


if __name__ == "__main__":
    observe()
