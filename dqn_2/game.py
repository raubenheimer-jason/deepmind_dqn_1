
# https://youtu.be/tsy1mgB7hB0?t=413

# includes meanings for different versions of breakout
# https://www.codeproject.com/Articles/5271947/Introduction-to-OpenAI-Gym-Atari-Breakout

# get num actions and action meanings
# https://stackoverflow.com/questions/63113154/how-to-check-out-actions-available-in-openai-gym-environment
# print(env.unwrapped.get_action_meanings())

# atari docs
# https://www.gymlibrary.ml/environments/atari/

# pip install gym[atari]
# pip install gym[accept-rom-license]

import gym
import numpy as np
import cv2

import torchvision
import torch


def init_game():
    print("init game")


class Preprocessing:
    def __init__(self, init_observation=None):
        """ 
        - raw Atari 2600 frames, which are 210 x 160 pixel images with a 128-colour palette
        - observation.shape = (210,160,3)
        - PyTorch order: (C, H, W)
        - observation.dtype = uint8
        - type(observation) = <class 'numpy.ndarray'>

        1: First, to encode a singleframe we take themaximum value for each pixel colour value over the frame being encoded and the previous frame. This was necessary to remove flickering that is present in games where some objects appear only in even frames while other objects appear only in odd frames, an artefact caused by the limited number of sprites Atari 2600 can display at once.

        2: Second, we then extract the Y channel, also known as luminance, from the RGB frame and rescale it to 84 x 84. The function phi from algorithm 1 described below applies this preprocessing to the m most recent frames and stacks them to produce the input to the Q-function, in which m = 4, although the algorithm is robust to different values of m (for example, 3 or 5).

        --> luminance, from the RGB --> just means make it greyscale??


        observation: this is the observation from the env, in the form of a numpy array (W,H,C)

        """

        self.calc_max = False if init_observation == None else True
        self.prev_obs = init_observation
        self.to_tensor = torchvision.transforms.ToTensor()
        self.transform = torch.nn.Sequential(
            # convert to grayscale
            # https://pytorch.org/vision/stable/generated/torchvision.transforms.Grayscale.html#torchvision.transforms.Grayscale
            torchvision.transforms.Grayscale(),

            # resize to 84x84
            # https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize
            torchvision.transforms.Resize([84, 84])

        )

    def process(self, observation):

        # max value of current and prev frame pix
        if self.calc_max:
            obs = np.maximum(observation, self.prev_obs)
            self.prev_obs = observation
            observation = obs

        # from H,W,C to C,H,W
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
        obs_t = self.to_tensor(observation)

        # compute transformations defined in __init__
        obs_t = self.transform(obs_t)

        return obs_t


def main():
    # n = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    # n = np.array([[[1, 2, 3], [4, 6, 5]]])
    n = np.array([[[1, 2, 3], [0, 0, 0]]])

    print(n.shape)
    # print(n.strides)

    # print(n[..., :3])
    n = np.dot(n[..., :3], [0.2989, 0.5870, 0.1140])  # rgb2gray
    # print(np.dot(n[..., :3], [0.2989, 0.5870, 0.1140]))
    print(n)
    print(n.shape)

    print(n[0, 0])

    # n = n.transpose((2, 0, 1))
    n = n.transpose((1, 0))

    # # print(np.dot(n[::3], [0.2989, 0.5870, 0.1140]))
    print(n.shape)
    # print(np.dot(n[3::], [0.2989, 0.5870, 0.1140]))

    # print("-----")

    # n = n.transpose((2, 0, 1))

    # print(n.shape)
    # # print(n.strides)


if __name__ == "__main__":
    main()
