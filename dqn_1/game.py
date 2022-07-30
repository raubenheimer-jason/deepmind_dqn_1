
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


def make_atari_env(env_id="ALE/Breakout-v5"):
    env = gym.make(env_id,
                   render_mode="rgb_array",  # or human
                   new_step_api=True)
    return env


def rgb2gray(rgb):
    """ Converts rgb np to single grayscale value 
    rgb: W,H,C

    https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# class Preprocessing(torch.nn.Module):
class Preprocessing:
    def __init__(self, init_observation):

        # super(Preprocessing, self).__init__()

        self.prev_obs = init_observation

        self.to_tensor = torchvision.transforms.ToTensor()

        # torchvision.transforms = torch.nn.Sequential(
        self.transform = torch.nn.Sequential(
            # convert to grayscale
            # https://pytorch.org/vision/stable/generated/torchvision.transforms.Grayscale.html#torchvision.transforms.Grayscale
            torchvision.transforms.Grayscale(),

            # resize to 84x84
            # https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize
            torchvision.transforms.Resize([84, 84])

        )
        # self.scripted_transforms = torch.jit.script(torchvision.transforms())

    def process(self, observation):

        # # max value of current and prev frame pix
        # if not self.prev_obs == None:
        #     # if prev_obs isn't None
        #     print("calc max")
        #     obs = np.maximum(observation, self.prev_obs)
        # else:
        #     obs = observation

        # max value of current and prev frame pix
        obs = np.maximum(observation, self.prev_obs)

        # obs = observation

        self.prev_obs = observation

        # from H,W,C to C,H,W
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
        obs_t = self.to_tensor(obs)

        # obs_t = torchvision.transforms.ToTensor(obs)

        # obs_t = torch.tensor(obs)
        # obs_t = self.scripted_transforms(obs_t)
        obs_t = self.transform(obs_t)

        return obs_t
        # return obs

    # def process(self, obs_1, obs_2):

    #     # from H,W,C to C,H,W
    #     # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
    #     obs_t = self.to_tensor(obs)

    #     # obs_t = torchvision.transforms.ToTensor(obs)

    #     # obs_t = torch.tensor(obs)
    #     obs_t = self.scripted_transforms(obs_t)
    #     # self.n(obs)

    #     return obs_t


def preprocessing_t(obs):
    """
    https://pytorch.org/vision/stable/transforms.html
    """

    obs_t = torchvision.transforms.ToTensor()
    ret = obs_t(obs)

    print(type(obs_t))
    print(type(ret))
    print(ret.shape)

    print(ret[:, 80, 80])

    return

    torchvision.transforms = torch.nn.Sequential(
        # from H,W,C to C,H,W
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
        torchvision.transforms.ToTensor(),

        # convert to grayscale
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.Grayscale.html#torchvision.transforms.Grayscale
        torchvision.transforms.Grayscale(),

        # resize to 84x84
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize
        torchvision.transforms.Resize([84, 84])

    )
    scripted_transforms = torch.jit.script(torchvision.transforms)
    # scripted_transforms(obs)


def preprocessing(observation):
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

    # convert from W,H,C to C,H,W
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html#numpy.ndarray.transpose
    observation = observation.transpose((2, 0, 1))  # converts to (C, H, W)#
    # observation = observation.transpose()  # converts to (C, H, W)#
    # observation = observation.transpose()  # converts to (C, H, W)#

    print(f"after transpose: {observation.shape}")

    obs_t = torch.tensor(observation)
    obs_t = torchvision.transforms.functional.rgb_to_grayscale(obs_t)

    obs_t = torch.reshape(obs_t, (1, 84, 84))

    observation = obs_t.numpy()

    print(observation.shape)

    # # convert to grayscale
    # observation = np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140])
    # # observation = np.dot(observation[3:, ...], [0.2989, 0.5870, 0.1140])
    # # observation = rgb2gray(observation)

    # print(f"after rgb2gray: {observation.shape}")

    # n = observation[80][200]
    # print(n)

    # # rescale image to 84x84 (from 160x210)
    # # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    # observation = cv2.resize(observation,
    #                          dsize=(84, 84),
    #                          interpolation=cv2.INTER_CUBIC)

    return observation


def preprocessing_old2(observation):
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

    # convert from W,H,C to C,H,W
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html#numpy.ndarray.transpose
    observation = observation.transpose((2, 0, 1))  # converts to (C, H, W)#
    # observation = observation.transpose()  # converts to (C, H, W)#
    # observation = observation.transpose()  # converts to (C, H, W)#

    print(f"after transpose: {observation.shape}")

    # # convert to grayscale
    # observation = np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140])
    # # observation = np.dot(observation[3:, ...], [0.2989, 0.5870, 0.1140])
    # # observation = rgb2gray(observation)

    # print(f"after rgb2gray: {observation.shape}")

    # n = observation[80][200]
    # print(n)

    # # rescale image to 84x84 (from 160x210)
    # # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    # observation = cv2.resize(observation,
    #                          dsize=(84, 84),
    #                          interpolation=cv2.INTER_CUBIC)

    return observation


def preprocessing_old(observation):
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

    # convert to grayscale
    observation = np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140])
    # observation = np.dot(observation[3:, ...], [0.2989, 0.5870, 0.1140])
    # observation = rgb2gray(observation)

    print(f"after rgb2gray: {observation.shape}")

    # convert from W,H,C to C,H,W
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html#numpy.ndarray.transpose
    # observation = observation.transpose((2, 0, 1))  # converts to (C, H, W)#
    # observation = observation.transpose()  # converts to (C, H, W)#
    observation = observation.transpose()  # converts to (C, H, W)#

    print(f"after transpose: {observation.shape}")

    # n = observation[80][200]
    # print(n)

    # # rescale image to 84x84 (from 160x210)
    # # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    # observation = cv2.resize(observation,
    #                          dsize=(84, 84),
    #                          interpolation=cv2.INTER_CUBIC)

    return observation


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
