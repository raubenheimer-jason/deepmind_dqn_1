
import pickle
import torch
import numpy as np
from constants import *
import random
import os


def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


# def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
def nature_cnn(n_stacked_frames, frame_size=(84, 84), depths=(32, 64, 64), final_layer=512):
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

    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(n_stacked_frames, depths[0], kernel_size=8, stride=4),
        torch.nn.ReLU(),
        torch.nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        torch.nn.ReLU(),
        torch.nn.Flatten()
    )

    dummy_stack = np.stack([np.stack(
        [np.zeros((frame_size[0], frame_size[1])) for _ in range(n_stacked_frames)])])

    # compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(dummy_stack).float()).shape[1]

    out = torch.nn.Sequential(cnn, torch.nn.Linear(
        n_flatten, final_layer), torch.nn.ReLU())

    return out


class Network(torch.nn.Module):
    """
    Separate output unit for each possible action, and only the state representation is an input to the neural network

    https://youtu.be/tsy1mgB7hB0?t=1563

    """

    def __init__(self, num_actions, n_stacked_frames):
        """
        Input:      84 x 84 x 4 image produced by the preprocessing map phi
        Output:     Single output for each valid action
        """
        super().__init__()

        self.num_actions = num_actions

        conv_net = nature_cnn(n_stacked_frames)

        self.net = torch.nn.Sequential(
            conv_net, torch.nn.Linear(512, self.num_actions))

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

        params = {k: torch.as_tensor(v, device=device)
                  for k, v in params_numpy.items()}

        self.load_state_dict(params)


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
        with torch.no_grad():
            phi_t_np = np.asarray([phi_t])
            phi_t_tensor = torch.as_tensor(
                phi_t_np, device=device, dtype=torch.float32)/255
            policy_q = policy_net(phi_t_tensor)
            max_q_index = torch.argmax(policy_q, dim=1)
            action = max_q_index.detach().item()

    return action


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

    phi_js_t = torch.as_tensor(phi_js, dtype=torch.float32, device=device)/255
    a_ts_t = torch.as_tensor(a_ts, dtype=torch.int64,
                             device=device).unsqueeze(-1)
    r_ts_t = torch.as_tensor(r_ts, dtype=torch.float32,
                             device=device).unsqueeze(-1)
    phi_jplus1s_t = torch.as_tensor(
        phi_jplus1s, dtype=torch.float32, device=device)/255
    dones_t = torch.as_tensor(dones, dtype=torch.uint8,
                              device=device).unsqueeze(-1)

    # compute targets
    target_q_values = target_net(phi_jplus1s_t)

    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
    # clever piecewise function (becasue if dones_t is 1 then targets just = rews_t)
    # maybe slow though because we calc max_target_q_values every time...?
    targets = r_ts_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Calc loss
    q_values = policy_net(phi_js_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=a_ts_t)
    loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)

    return loss
