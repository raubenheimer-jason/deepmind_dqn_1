# used to see the model play the game

from main import Network, select_action
import gym
import torch
import itertools
import time


def observe():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    env = gym.make("ALE/Breakout-v5",
                   render_mode="human",
                   new_step_api=True)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4, new_step_api=True)
    #! need to add "max pix value" to observation...
    num_actions = env.action_space.n
    env_obs_space = env.observation_space

    net = Network(num_actions, env_obs_space).to(device)

    net.load("../models/2022-08-02__11-52-56/300k.pkl", device)
    print("done loading")

    obs = env.reset()
    beginning_episode = True
    for t in itertools.count():
        # step=-1 so we never select a random action
        action = select_action(num_actions, -1, obs, net, device)

        if beginning_episode:
            action = 1  # "FIRE"
            beginning_episode = False

        obs, _, term, trun, _ = env.step(action)
        time.sleep(0.02)

        if term or trun:
            obs = env.reset()
            beginning_episode = True


if __name__ == "__main__":
    observe()
