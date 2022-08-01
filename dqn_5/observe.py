# used to see the model play the game

from dqn import Network, select_action
# import os
import gym
# import random
# import numpy as np
import torch
# from torch import nn
import itertools
# from baselines_wrappers import DummyVecEnv
# from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import time

# import msgpack
# from msgpack_numpy import patch as msgpack_numpy_patch
# msgpack_numpy_patch()


# def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
#     n_input_channels = observation_space.shape[0]

#     cnn = nn.Sequential(
#         nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
#         nn.ReLU(),
#         nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
#         nn.ReLU(),
#         nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
#         nn.ReLU(),
#         nn.Flatten())

#     # Compute shape by doing one forward pass
#     with torch.no_grad():
#         n_flatten = cnn(torch.as_tensor(
#             observation_space.sample()[None]).float()).shape[1]

#     out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

#     return out


# class Network(nn.Module):
#     def __init__(self, env, device):
#         super().__init__()

#         self.num_actions = env.action_space.n
#         self.device = device

#         conv_net = nature_cnn(env.observation_space)

#         self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

#     def forward(self, x):
#         return self.net(x)

#     def act(self, obses, epsilon):
#         obses_t = torch.as_tensor(
#             obses, dtype=torch.float32, device=self.device)
#         q_values = self(obses_t)

#         max_q_indices = torch.argmax(q_values, dim=1)
#         actions = max_q_indices.detach().tolist()

#         for i in range(len(actions)):
#             rnd_sample = random.random()
#             if rnd_sample <= epsilon:
#                 actions[i] = random.randint(0, self.num_actions - 1)

#         return actions

#     def load(self, load_path):
#         if not os.path.exists(load_path):
#             raise FileNotFoundError(load_path)

#         with open(load_path, 'rb') as f:
#             params_numpy = msgpack.loads(f.read())

#         params = {k: torch.as_tensor(v, device=self.device)
#                   for k, v in params_numpy.items()}

#         self.load_state_dict(params)


def observe():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # def make_env(): return make_atari_deepmind(
    #     'BreakoutNoFrameskip-v4', scale_values=True, render_mode="human")

    env = gym.make("ALE/Breakout-v5",
                   render_mode="human",
                   new_step_api=True)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4, new_step_api=True)
    #! need to add "max pix value" to observation...
    num_actions = env.action_space.n
    env_obs_space = env.observation_space

    # vec_env = DummyVecEnv([make_env for _ in range(1)])

    # env = BatchedPytorchFrameStack(vec_env, k=4)

    net = Network(num_actions, env_obs_space).to(device)

    # net = Network(env, device)
    # net = net.to(device)

    # net.load('./atari_model.pack')
    # net.load("../models/2022-08-01__10-04-20.pkl", device)
    # net.load("../models/2022-08-01__11-24-20/_100k.pkl", device)
    # net.load("../models/2022-08-01__16-05-09/300k.pkl", device)
    net.load("../models/2022-08-01__18-01-36/300k.pkl", device)
    print("done loading")

    obs = env.reset()
    beginning_episode = True
    for t in itertools.count():
        # if isinstance(obs[0], PytorchLazyFrames):
        #     act_obs = np.stack([o.get_frames() for o in obs])
        #     action = net.act(act_obs, 0.0)
        # else:
        #     action = net.act(obs, 0.0)

        # step=-1 so we never select a random action
        action = select_action(num_actions, -1, obs, net, device)

        # print(f"action: {action}")

        if beginning_episode:
            action = 1  # "FIRE"
            beginning_episode = False

        obs, _, term, trun, _ = env.step(action)
        # env.render()
        time.sleep(0.02)

        if term or trun:
            obs = env.reset()
            beginning_episode = True


if __name__ == "__main__":
    observe()
    # print("this can run at the same time")
    # from os import walk

    # filenames = next(walk("../models"), (None, None, []))[2]  # [] if no file

    # print(filenames)
