
import os
import random
import torch
import torch.nn as nn

import pickle

import numpy as np


INITIAL_EXPLORATION = 1  # Initial value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION = 0.1  # final value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION_FRAME = int(1e6)  # num frames epsilon changes linearly

REPLAY_START_SIZE = int(5e4)  # uniform random policy run before learning
# REPLAY_START_SIZE = 35  # uniform random policy run before learning


# DECAY_SLOPE = (1.0-0.1)/(0.0-1e6)
# delta y over delta x
DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
    (REPLAY_START_SIZE-(REPLAY_START_SIZE+FINAL_EXPLORATION_FRAME))
DECAY_C = INITIAL_EXPLORATION - (DECAY_SLOPE*REPLAY_START_SIZE)
# DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
#     (0.0-FINAL_EXPLORATION_FRAME)


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
    """
    CHW format (channels, height, width)

    Input:        84 x 84 x 4 image produced by the preprocessing map phi
    1st hidden:   Convolves 32 filters of 8 x 8 with stride 4 with the input image and applies a rectifier nonlinearity
    2nd hidden:   Convolves 64 filters of 4 x 4 with stride 2, and applies a rectifier nonlinearity
    3rd hidden:   Convolves 64 filters of 3 x 3 with stride 1, and applies a rectifier nonlinearity
    4th hidden: Fully-connected and consists of 512 rectifier units.
    Output:       Fully-connected linear layer with a single output for each valid action (varied between 4-18 in the games considered)

    https://youtu.be/tsy1mgB7hB0?t=1563

    Replicating the algorithm in the paper: Human-level control through deep reinforcement learning. Under: Methods, Model architecture
    """
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten()
    )

    # compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(
            observation_space.sample()[None]).float()).shape[1]

    # print(f"n_flatten: {n_flatten}")

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out


class Network(nn.Module):
    """
    Separate output unit for each possible action, and only the state representation is an input to the neural network

    https://youtu.be/tsy1mgB7hB0?t=1563

    """

    def __init__(self, num_actions, env_obs_space):
        """ 
        Input:      84 x 84 x 4 image produced by the preprocessing map phi
        Output:     Single output for each valid action
        """
        # super(Network, self).__init__()
        super().__init__()

        self.num_actions = num_actions
        self.env_obs_space = env_obs_space

        conv_net = nature_cnn(env_obs_space)

        # print(f"self.num_actions: {self.num_actions}")

        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

    def forward(self, x):
        # print(x)
        # print(type(x))
        # print(f"x.shape: {x.shape}")
        return self.net(x)

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy()
                  for k, t in self.state_dict().items()}
        # params_data = msgpack.dumps(params)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            # f.write(params_data)
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, load_path, device):
        # if not os.path.exists(load_path):
        #     raise FileNotFoundError(load_path)

        # with open(load_path, 'rb') as f:
        #     params_numpy = msgpack.loads(f.read())

        with open(load_path, 'rb') as f:
            params_numpy = pickle.load(f)

        params = {k: torch.as_tensor(v, device=device)
                  for k, v in params_numpy.items()}

        self.load_state_dict(params)


def select_action(num_actions, step, phi_t, policy_net, device):
    """ selects action, either random or from model """

    # epsilon = np.interp(self.step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    # epsilon = np.interp(self.step,
    #                     [0, FINAL_EXPLORATION_FRAME],
    #                     [INITIAL_EXPLORATION, FINAL_EXPLORATION])

    # # y=mx+c to for step<=FINAL_EXPLORATION_FRAME else epsilon=FINAL_EXPLORATION
    # epsilon = DECAY_SLOPE*step + \
    #     1 if step <= FINAL_EXPLORATION_FRAME else FINAL_EXPLORATION

    if step > (REPLAY_START_SIZE + FINAL_EXPLORATION_FRAME):
        # if step > (5e4 + 1e6)
        epsilon = FINAL_EXPLORATION

    elif step > REPLAY_START_SIZE:
        # step must be <= (5e4 + 1e6) but greater than 5e4
        # slope part of epsilon
        # see pdf paper notes bottom of page 6 for working
        epsilon = DECAY_SLOPE*step + DECAY_C

    elif step >= 0:
        # step must be <= 5e4, still in initialise replay mem state
        # setting epsilon = 1 ensures that we always choose a random action
        # random.random --> the interval [0, 1), which means greater than or equal to 0 and less than 1
        epsilon = 1

    else:
        # this is for when step=-1 is passed,
        # used for the "observe.py" script where we always want the model to select the action
        # (no random action selection)
        epsilon = -1

    # else:
    #     # step must be <= 5e4, still in initialise replay mem state
    #     # setting epsilon = 1 ensures that we always choose a random action
    #     # random.random --> the interval [0, 1), which means greater than or equal to 0 and less than 1
    #     epsilon = 1

    rand_sample = random.random()

    # if step >= REPLAY_START_SIZE:
    #     print(f"epsilon: {epsilon}, rand_sample: {rand_sample}")

    if rand_sample < epsilon:
        action = random.randrange(num_actions)
        # return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
        # print(f"random action: {action}")
    else:
        with torch.no_grad():
            # print("select action ------------------------------")
            # convert phi_t to tensor
            phi_t = np.asarray(phi_t)
            phi_t_tensor = torch.as_tensor(
                phi_t, device=device, dtype=torch.float32)
            phi_t_tensor = torch.stack([phi_t_tensor])
            policy_q = policy_net(phi_t_tensor)
            # max_q_indices = torch.argmax(policy_q, dim=1)
            max_q_index = torch.argmax(policy_q, dim=1)
            # actions = max_q_indices.detach().tolist()
            action = max_q_index.detach().item()
            # return self.policy_net(state).max(1)[1].view(1, 1)
            # print(f"policy_q action: {action}")

            # print(f"action: {action}")

    return action

    # def select_action(self, step, phi_t):
    #     """ selects action, either random or from model """

    #     # epsilon = np.interp(self.step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    #     # epsilon = np.interp(self.step,
    #     #                     [0, FINAL_EXPLORATION_FRAME],
    #     #                     [INITIAL_EXPLORATION, FINAL_EXPLORATION])

    #     # y=mx+c to for step<=FINAL_EXPLORATION_FRAME else epsilon=FINAL_EXPLORATION
    #     epsilon = DECAY_SLOPE*step + \
    #         1 if step <= FINAL_EXPLORATION_FRAME else FINAL_EXPLORATION

    #     if random.random() < epsilon:
    #         action = random.randrange(self.num_actions)
    #         # return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
    #     else:
    #         with torch.no_grad():
    #             action = "action from Q network..."
    #             policy_q = policy_net(phi_t)
    #             # return self.policy_net(state).max(1)[1].view(1, 1)

    #     return action

    # def calc_y_j(self, minibatch, target_net):
    #     """ calculates targets y_j

    #         y_j = r_j if episode terminates at step j+1
    #         otherwise
    #         y_j = r_j + gamma * "max_target_q_values"

    #         minibatch = batch of transitions (phi_t, a_t, r_t, phi_tplus1, done)

    #     """

    #     y_j = []

    #     for transition in minibatch:
    #         r_j = transition[2]

    #         if transition[4] == True:
    #             # done == true
    #             y_j.append(r_j)
    #         else:
    #             target_q_values = target_net(new_obses_t)
    #             max_target_q =
    #             y_j_val = r_j + max_target_q
