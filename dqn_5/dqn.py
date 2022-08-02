
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
# REPLAY_START_SIZE = 5000  # uniform random policy run before learning #! testing

GAMMA = 0.99  # discount factor used in Q-learning update

# # delta y over delta x
# DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
#     (0-FINAL_EXPLORATION_FRAME)
# DECAY_C = INITIAL_EXPLORATION

# delta y over delta x
DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
    (REPLAY_START_SIZE-(REPLAY_START_SIZE+FINAL_EXPLORATION_FRAME))
DECAY_C = INITIAL_EXPLORATION - (DECAY_SLOPE*REPLAY_START_SIZE)


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


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

#! this is my version
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

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out


# class Network(nn.Module):
#     def __init__(self, env, device):
#         super().__init__()

#         self.num_actions = env.action_space.n
#         self.device = device

#         conv_net = nature_cnn(env.observation_space)

#         self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

#     def forward(self, x):
#         return self.net(x)

#     def compute_loss(self, transitions, target_net):
#         obses = [t[0] for t in transitions]
#         actions = np.asarray([t[1] for t in transitions])
#         rews = np.asarray([t[2] for t in transitions])
#         new_obses = [t[3] for t in transitions]
#         dones = np.asarray([t[4] for t in transitions])

#         # print(f"type(obses): {type(obses)}")
#         # print(f"len(obses): {len(obses)}")

#         # if isinstance(obses[0], PytorchLazyFrames):
#         #     obses = np.stack([o.get_frames() for o in obses])
#         #     new_obses = np.stack([o.get_frames() for o in new_obses])
#         #     # print("isinstance(obses[0], PytorchLazyFrames)")
#         # else:
#         obses = np.asarray(obses)
#         new_obses = np.asarray(new_obses)
#         # print("other...............")

#         # print(f"type(obses) 2: {type(obses)}")
#         # print(f"obses.shape 2: {obses.shape}")
#         # print(f"type(obses[0]) 2: {type(obses[0])}")
#         # print(f"obses[0].shape 2: {obses[0].shape}")

#         # print("---")

#         # print(f"type(obses[0][0]) 2: {type(obses[0][0])}")
#         # print(f"obses[0][0].shape 2: {obses[0][0].shape}")

#         # print("----------------------------")

#         obses_t = torch.as_tensor(
#             obses, dtype=torch.float32, device=self.device)
#         actions_t = torch.as_tensor(
#             actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
#         rews_t = torch.as_tensor(
#             rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
#         dones_t = torch.as_tensor(
#             dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
#         new_obses_t = torch.as_tensor(
#             new_obses, dtype=torch.float32, device=self.device)

#         # Compute Targets
#         target_q_values = target_net(new_obses_t)
#         print(f"target_q_values: {target_q_values}")

#         max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

#         # print(f"max_target_q_values: {max_target_q_values}")

#         # clever piecewise function (becasue if dones_t is 1 then targets just = rews_t)
#         # maybe slow though because we calc max_target_q_values every time...
#         targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

#         # Compute Loss
#         q_values = self(obses_t)

#         action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

#         loss = nn.functional.smooth_l1_loss(action_q_values, targets)

#         return loss


#! this is my version
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
        super().__init__()

        self.num_actions = num_actions
        self.env_obs_space = env_obs_space

        conv_net = nature_cnn(env_obs_space)

        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

    def forward(self, x):
        return self.net(x)

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy()
                  for k, t in self.state_dict().items()}

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, load_path, device):
        with open(load_path, 'rb') as f:
            params_numpy = pickle.load(f)

        # print(f"params_numpy: {params_numpy}")

        params = {k: torch.as_tensor(v, device=device)
                  for k, v in params_numpy.items()}

        self.load_state_dict(params)


# def select_action(num_actions, step, phi_t, policy_net, device):
#     """ selects action, either random or from model """

#     if step > FINAL_EXPLORATION_FRAME:
#         # if step > 1e6
#         epsilon = FINAL_EXPLORATION

#     elif step >= 0:
#         # step must be <= 1e6 but greater than 5e4
#         # slope part of epsilon
#         # see pdf paper notes bottom of page 6 for working
#         epsilon = DECAY_SLOPE*step + DECAY_C

#     else:
#         # this is for when step=-1 is passed,
#         # used for the "observe.py" script where we always want the model to select the action
#         # (no random action selection)
#         epsilon = -1

#     rand_sample = random.random()
#     if rand_sample < epsilon:
#         action = random.randrange(num_actions)
#     else:
#         # print(f"type(phi_t): {type(phi_t)}")
#         # print(f"phi_t.shape: {phi_t.shape}")
#         # with torch.no_grad():
#         # convert phi_t to tensor
#         # phi_t = np.asarray(phi_t)
#         phi_t = np.asarray([phi_t])
#         # print(f"type(phi_t) 2: {type(phi_t)}")
#         # print(f"phi_t.shape 2: {phi_t.shape}")
#         phi_t_tensor = torch.as_tensor(
#             phi_t, device=device, dtype=torch.float32)
#         # phi_t_tensor = torch.stack([phi_t_tensor])
#         policy_q = policy_net(phi_t_tensor)
#         max_q_index = torch.argmax(policy_q, dim=1)
#         action = max_q_index.detach().item()

#     # print(action)

#     return action


def select_action(num_actions, step, phi_t, policy_net, device):
    """ selects action, either random or from model """

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

    rand_sample = random.random()
    if rand_sample < epsilon:
        action = random.randrange(num_actions)
    else:
        # print(f"type(phi_t): {type(phi_t)}")
        # print(f"phi_t.shape: {phi_t.shape}")
        # with torch.no_grad():
        # convert phi_t to tensor
        # phi_t = np.asarray(phi_t)

        # # convert to C, H, W
        # _image = np.array(phi_t)
        # image = torch.from_numpy(_image)
        # phi_t_tensor = image[np.newaxis, :]

        phi_t_np = np.asarray([phi_t])
        # print(f"type(phi_t) 2: {type(phi_t)}")
        # print(f"phi_t.shape 2: {phi_t.shape}")
        phi_t_tensor = torch.as_tensor(
            phi_t_np, device=device, dtype=torch.float32)

        phi_t_tensor = torch.div(phi_t_tensor, 255)

        # phi_t_tensor = torch.stack([phi_t_tensor])
        # phi_t_tensor = phi_t_tensor.permute(0, 3, 1, 2)

        policy_q = policy_net(phi_t_tensor)
        max_q_index = torch.argmax(policy_q, dim=1)
        action = max_q_index.detach().item()

    # print(action)

    return action


# def select_action(num_actions, step, phi_t, policy_net, device):
#     """ selects action, either random or from model """

#     if step > (REPLAY_START_SIZE + FINAL_EXPLORATION_FRAME):
#         # if step > (5e4 + 1e6)
#         epsilon = FINAL_EXPLORATION

#     elif step > REPLAY_START_SIZE:
#         # step must be <= (5e4 + 1e6) but greater than 5e4
#         # slope part of epsilon
#         # see pdf paper notes bottom of page 6 for working
#         epsilon = DECAY_SLOPE*step + DECAY_C

#     elif step >= 0:
#         # step must be <= 5e4, still in initialise replay mem state
#         # setting epsilon = 1 ensures that we always choose a random action
#         # random.random --> the interval [0, 1), which means greater than or equal to 0 and less than 1
#         epsilon = 1

#     else:
#         # this is for when step=-1 is passed,
#         # used for the "observe.py" script where we always want the model to select the action
#         # (no random action selection)
#         epsilon = -1

#     rand_sample = random.random()
#     if rand_sample < epsilon:
#         action = random.randrange(num_actions)
#     else:
#         with torch.no_grad():
#             # convert phi_t to tensor
#             phi_t = np.asarray(phi_t)
#             phi_t_tensor = torch.as_tensor(phi_t, device=device, dtype=torch.float32)
#             phi_t_tensor = torch.stack([phi_t_tensor])
#             policy_q = policy_net(phi_t_tensor)
#             max_q_index = torch.argmax(policy_q, dim=1)
#             action = max_q_index.detach().item()

#     return action


def calc_loss(minibatch, target_net, policy_net, device):
    """ calculates loss: (y_j - Q(phi_j, a_j; theta))^2

        calculating targets y_j:
        y_j = r_j if episode terminates at step j+1
        otherwise
        y_j = r_j + gamma * "max_target_q_values"

        minibatch = batch of transitions (phi_t, a_t, r_t, phi_tplus1, done)

    """

    phi_js = np.asarray([t[0] for t in minibatch])
    a_ts = np.asarray([t[1] for t in minibatch])
    r_ts = np.asarray([t[2] for t in minibatch])
    phi_jplus1s = np.asarray([t[3] for t in minibatch])
    dones = np.asarray([t[4] for t in minibatch])

    # print(f"type(phi_js): {type(phi_js)}")
    # print(f"len(phi_js): {len(phi_js)}")

    # print(f"type(phi_js) 2: {type(phi_js)}")
    # print(f"phi_js.shape 2: {phi_js.shape}")
    # print(f"type(phi_js[0]) 2: {type(phi_js[0])}")
    # print(f"phi_js[0].shape 2: {phi_js[0].shape}")

    # print("---")

    # print(f"type(phi_js[0][0]) 2: {type(phi_js[0][0])}")
    # print(f"phi_js[0][0].shape 2: {phi_js[0][0].shape}")

    # print("----------------------------")

    # # convert to C, H, W
    # _image = np.array(phi_js)
    # image = torch.from_numpy(_image)
    # phi_js_t = image[np.newaxis, :]

    # # convert to C, H, W
    # _image = np.array(phi_jplus1s)
    # image = torch.from_numpy(_image)
    # phi_jplus1s_t = image[np.newaxis, :]

    phi_js_t = torch.as_tensor(phi_js, dtype=torch.float32, device=device)

    # scale greyscale to between 0 and 1 (inclusive)
    phi_js_t = torch.div(phi_js_t, 255)

    # phi_js_t = phi_js_t.permute(0, 3, 1, 2)
    # phi_js_t = phi_js_t.permute(0, 1, 4, 2, 3)
    a_ts_t = torch.as_tensor(a_ts, dtype=torch.int64,
                             device=device).unsqueeze(-1)
    r_ts_t = torch.as_tensor(r_ts, dtype=torch.float32,
                             device=device).unsqueeze(-1)
    phi_jplus1s_t = torch.as_tensor(
        phi_jplus1s, dtype=torch.float32, device=device)
    # phi_jplus1s_t = phi_jplus1s_t.permute(0, 1, 4, 2, 3)

    phi_jplus1s_t = torch.div(phi_jplus1s_t, 255)


    dones_t = torch.as_tensor(
        dones, dtype=torch.float32, device=device).unsqueeze(-1)

    # print(f"new_obses_t: {phi_jplus1s_t}")

    # compute targets
    target_q_values = target_net(phi_jplus1s_t)
    # print(f"target_q_values: {target_q_values}")
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    # print(f"max_target_q_values: {max_target_q_values}")

    # clever piecewise function (becasue if dones_t is 1 then targets just = rews_t)
    # maybe slow though because we calc max_target_q_values every time...
    targets = r_ts_t + GAMMA * (1 - dones_t) * max_target_q_values

    # print(f"targets: {targets}")

    # Calc loss
    q_values = policy_net(phi_js_t)
    # print(q_values)
    action_q_values = torch.gather(input=q_values, dim=1, index=a_ts_t)
    loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)

    # print(f"loss: {loss}")

    return loss
