"""
Changes:
-> Using BreakoutNoFrameskip-v4

# TODO
-> Loss of a life sets "done" flag to true (otherwise no penalty for loosing a life)
-> manual "frame skip" where it takes the max over two frames as well...
-> "states" need to overlap: s1 = {x1, x2, x3, x4}, s2 = {x2, x3, x4, x5} (assuming that x1->x2 has already taken frame skipping into account...)
"""


import pickle
import torch.nn as nn
import os
from collections import deque
import random
import torch
# from game import Preprocessing
import numpy as np
import gym
from itertools import count
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

AGENT_HISTORY_LEN = 4  # Number of most recent frames given as input to the Q network
UPDATE_FREQ = 4  # Agent selects 4 actions between each pair of successive updates
NO_OP_MAX = 30  # max num of "do nothing" actions performed by agent at the start of an episode

FRAMES_SKIP = 4  # I added this...

REPLAY_MEM_SIZE = int(1e6)
BATCH_SIZE = 32
LEARNING_RATE = 0.25e-3  # learning rate used by RMSProp
# LEARNING_RATE = 5e-5  # learning rate used by RMSProp <<-- from youtube dude...
GRADIENT_MOMENTUM = 0.95  # RMSProp
SQUARED_GRADIENT_MOMENTUM = 0.95  # RMSProp
MIN_SQUARED_GRADIENT = 0.01  # RMSProp
TARGET_NET_UPDATE_FREQ = int(1e4)  # C
ACTION_REPEAT = 4  # Agent only sees every 4th input frame (repeat last action)
PRINT_INFO_FREQ = int(1e3)

LOG_DIR = "./logs/"
LOG_INTERVAL = 1000

SAVE_DIR = "./models/"
SAVE_INTERVAL = 10000
SAVE_NEW_FILE_INTERVAL = int(1e5)

LOGGING = False
PRINTING = True
SAVING = False


INITIAL_EXPLORATION = 1  # Initial value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION = 0.1  # final value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION_FRAME = int(1e6)  # num frames epsilon changes linearly

REPLAY_START_SIZE = int(5e4)  # uniform random policy run before learning
# REPLAY_START_SIZE = 5000  # uniform random policy run before learning #! testing

GAMMA = 0.99  # discount factor used in Q-learning update


# delta y over delta x
DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
    (REPLAY_START_SIZE-(REPLAY_START_SIZE+FINAL_EXPLORATION_FRAME))
DECAY_C = INITIAL_EXPLORATION - (DECAY_SLOPE*REPLAY_START_SIZE)


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
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
    # n_input_channels = observation_space.shape[0]

    # print(observation_space.shape)

    # print(f"n_input_channels: {n_input_channels}")

    cnn = nn.Sequential(
        # nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.Conv2d(n_stacked_frames, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten()
    )

    # need it to look like: (1, 4, 84, 84)

    # dummy_stack = np.stack([observation_space.sample()[None] for _ in range(4)])
    dummy_stack = np.stack([np.stack(
        [np.zeros((frame_size[0], frame_size[1])) for _ in range(n_stacked_frames)])])

    # print(dummy_stack.shape) # (1, 4, 84, 84)
    # print(torch.as_tensor(dummy_stack).float().shape) # torch.Size([1, 4, 84, 84])
    # print(torch.as_tensor(dummy_stack).float())
    # print(torch.as_tensor(dummy_stack).float().dtype)

    # print(torch.as_tensor(observation_space.sample()[None]).float().shape) # torch.Size([1, 4, 84, 84])

    # print(observation_space.sample()[None].shape)

    # print(torch.as_tensor(observation_space.sample()[None]).float())
    # print(torch.as_tensor(observation_space.sample()[None]).float().dtype)

    # compute shape by doing one forward pass
    with torch.no_grad():
        # n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        n_flatten = cnn(torch.as_tensor(dummy_stack).float()).shape[1]

        # print(f"n_flatten: {n_flatten}")

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out


class Network(nn.Module):
    """
    Separate output unit for each possible action, and only the state representation is an input to the neural network

    https://youtu.be/tsy1mgB7hB0?t=1563

    """

    # def __init__(self, num_actions, env_obs_space):
    def __init__(self, num_actions, n_stacked_frames):
        """
        Input:      84 x 84 x 4 image produced by the preprocessing map phi
        Output:     Single output for each valid action
        """
        super().__init__()

        self.num_actions = num_actions
        # self.env_obs_space = env_obs_space

        # conv_net = nature_cnn(env_obs_space)
        conv_net = nature_cnn(n_stacked_frames)

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
        phi_t_np = np.asarray([phi_t])
        phi_t_tensor = torch.as_tensor(
            phi_t_np, device=device, dtype=torch.float32)
        phi_t_tensor = torch.div(phi_t_tensor, 255)
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

    phi_js_t = torch.as_tensor(phi_js, dtype=torch.float32, device=device)
    # scale greyscale to between 0 and 1 (inclusive)
    phi_js_t = torch.div(phi_js_t, 255)
    a_ts_t = torch.as_tensor(a_ts, dtype=torch.int64,
                             device=device).unsqueeze(-1)
    r_ts_t = torch.as_tensor(r_ts, dtype=torch.float32,
                             device=device).unsqueeze(-1)
    phi_jplus1s_t = torch.as_tensor(
        phi_jplus1s, dtype=torch.float32, device=device)
    phi_jplus1s_t = torch.div(phi_jplus1s_t, 255)
    dones_t = torch.as_tensor(
        dones, dtype=torch.float32, device=device).unsqueeze(-1)

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


class State:
    def __init__(self):
        """
        needs to take in every "observation", after every "step"
        - the action passed into step might be a repeat of the previous action, that is already taken care of.

        - this class needs to construct the "current state"
        -> only "store" every 4th frame (but also store every 3rd to get the max pixel values...)

        -> method "get_state" returns eg: s1 = {x1, x2, x3, x4}
            where x1 has already taken the max between two frames
            and x1 to x2 has already taken into account the skipping

        https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        """

        n_frames = 4
        frame_h = 84
        frame_w = 84

        self.x_buffer = deque(maxlen=n_frames)  # stores x1, x2, x3, x4
        for _ in range(n_frames):
            # initialise with black frames (0)
            n = np.zeros(shape=(frame_h, frame_w))
            self.x_buffer.append(n)

        # self.curr_frame = np.zeros(shape=(frame_h, frame_w))
        self.prev_frame = np.zeros(shape=(frame_h, frame_w))

        self.frame_count = 0
        self.frame_skip = 4

    def add_frame(self, frame):
        """ add a frame (74, 84) to the deque ONLY if we have skipped enough...

        """

        # increment frame count
        self.frame_count += 1

        # print(f"frame: {frame}")
        # print(f"frame.shape: {frame.shape}")

        # if frame count = 4 and frame skip = 4 then this is the frame we need to store
        if self.frame_count % self.frame_skip == 0:
            # but we need to look at the prev frame to get the max pixel values...
            curr_frame = np.maximum(frame, self.prev_frame)
            # now we store this frame in the deque
            self.x_buffer.append(curr_frame)

        elif (self.frame_count+1) % self.frame_skip == 0:
            # only need to store the frame just before we're about the sample the 4th frame...
            # set frame to prev_frame
            self.prev_frame = frame

    def get_state(self):
        """ returns s1 = {x1, x2, x3, x4}
            in the shape: (4, 84, 84)

            This gets called each iteration...

            as numpy array?
        """

        return np.stack(self.x_buffer)


def main():
    now = datetime.now()  # current date and time
    time_str = now.strftime("%Y-%m-%d__%H-%M-%S")
    log_path = LOG_DIR + time_str
    save_dir = f"{SAVE_DIR}{time_str}/"  # different folder for each "run"
    if LOGGING:
        summary_writer = SummaryWriter(log_path)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # game holds the env
    # env = gym.make("ALE/Breakout-v5",
    env = gym.make("BreakoutNoFrameskip-v4",
                   render_mode="rgb_array",  # or human
                   new_step_api=True)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    # env = gym.wrappers.FrameStack(env, 4, new_step_api=True)
    #! need to add "max pix value" to observation...
    num_actions = env.action_space.n
    # env_obs_space = env.observation_space

    # * Initialize replay memory D to capacity N
    replay_mem = deque(maxlen=REPLAY_MEM_SIZE)  # replay_mem is D
    # Need to fill the replay_mem (to REPLAY_START_SIZE) with the results from random actions
    #   -> maybe do this in the main loop and just select random until len(replay_mem) >= REPLAY_START_SIZE

    # * Initialize action-value function Q with random weights Theta
    # initialise policy_net
    # policy_net = Network(num_actions, env_obs_space).to(device)
    policy_net = Network(num_actions, AGENT_HISTORY_LEN).to(device)
    policy_net.apply(init_weights)

    # * Initialize target action-value function Q_hat with weights Theta_bar = Theta
    # initialise target_net
    # target_net = Network(num_actions, env_obs_space).to(device)
    target_net = Network(num_actions, AGENT_HISTORY_LEN).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # # # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    # optimiser = torch.optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE, alpha=0.99,
    #                                 eps=1e-08, weight_decay=0, momentum=GRADIENT_MOMENTUM, centered=False, foreach=None)
    optimiser = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    step = 0

    episode_rewards = []
    episode_lengths = []

    rewards_buffer = deque([], maxlen=100)
    lengths_buffer = deque([], maxlen=100)

    a_t = 0  # defined here to do the "frame skipping"

    prev_life = 0

    # obs_num = 0

    state = State()

    # * For episode = 1, M do
    for episode in count():

        episode_rewards.append(0.0)
        episode_lengths.append(0)

        # * Initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)
        # phi_t=1, preprocessed sequence
        if prev_life == 0:
            # only reset if lost all lives
            # phi_t, info = env.reset(return_info=True)
            frame, info = env.reset(return_info=True)
            state.add_frame(frame)
            phi_t = state.get_state()
            prev_life = info['lives']

        # * For t = 1, T do
        for t in count():
            step += 1

            # * With probability epsilon select a random action a_t
            # * otherwise select a_t = argmax_a Q(phi(s_t),a;Theta)
            if step % ACTION_REPEAT == 0:
                # "frame-skipping" technique where agent only selects a new action on every kth frame.
                # running step requires a lot less computation than having the agent select action
                # this allows roughly k times more games to be played without significantly increasing runtime
                a_t = select_action(num_actions, step,
                                    phi_t, policy_net, device)

            # * Execute action a_t in emulator and observe reward r_t and image x_t+1
            # phi_tplus1, r_t, term, trun, info = env.step(a_t)  # x_tplus1
            new_frame, r_t, term, trun, info = env.step(a_t)

            # * Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)
            state.add_frame(new_frame)
            # get new state
            phi_tplus1 = state.get_state()

            # obs_num += 1

            # if lost a life then mark the end of an episode so the agent gets a negative result for loosing a life
            if not prev_life == info['lives']:
                # print("lost a life")
                lost_life = True
                prev_life = info['lives']
            else:
                # print("DIDNT loose a life")
                lost_life = False

            # print(f"type(info['lives']): {type(info['lives'])}")
            # print(f"info['lives']: {info['lives']}")
            # print(f"obs_num: {obs_num}\tinfo: {info}")

            # done flag (terminated or truncated or lost_life)
            done_tplus1 = term or trun or lost_life

            episode_rewards[episode] += r_t
            episode_lengths[episode] += 1

            # * Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)

            # * Store transition (phi_t, a_t, r_t, phi_t+1) in D
            # added done flag (tplus1 to matach phi_tplus1)
            transition = (phi_t, a_t, r_t, phi_tplus1, done_tplus1)
            replay_mem.append(transition)  # replay_mem is D

            phi_t = phi_tplus1

            # don't take minibatch until replay mem has been initialised
            if step > REPLAY_START_SIZE:
                # * Sample random minibatch of transitions (phi_j, a_j, r_j, phi_j+1) from D
                minibatch = random.sample(replay_mem, BATCH_SIZE)

                # * Set y_j = r_j if episode terminates at step j+1
                # * otherwise set y_j = r_j + gamma * max_a_prime Q_hat(phi_j+1, a_prime; Theta_bar)
                # * Perform a gradient descent step on (y_j - Q(phi_j, a_j; Theta))^2 with respect to the network parameters Theta

                # calculate loss [ (y_j - Q(phi_j, a_j; Theta))^2 ]
                loss = calc_loss(minibatch, target_net, policy_net, device)

                # Gradient Descent
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # * Every C steps reset Q_hat = Q
                # Update Target Network
                if step % TARGET_NET_UPDATE_FREQ == 0:
                    # print(f"{step} --> update target net")
                    target_net.load_state_dict(policy_net.state_dict())

            # Logging
            if (LOGGING or PRINTING) and step % LOG_INTERVAL == 0:
                rew_mean = np.mean(rewards_buffer) or 0
                len_mean = np.mean(lengths_buffer) or 0

                if PRINTING:
                    print()
                    print('Step', step)
                    print('Avg Rew (mean last 100 episodes)', rew_mean)
                    print('Avg Ep steps (mean last 100 episodes)', len_mean)
                    print('Episodes', episode)

                if LOGGING:
                    summary_writer.add_scalar(
                        'AvgRew', rew_mean, global_step=step)
                    summary_writer.add_scalar(
                        'AvgEpLen', len_mean, global_step=step)
                    summary_writer.add_scalar(
                        'Episodes', episode, global_step=step)

            # Save
            if SAVING and step % SAVE_INTERVAL == 0 and step >= SAVE_NEW_FILE_INTERVAL:
                print('Saving...')
                # every 100k steps save a new version
                if step % SAVE_NEW_FILE_INTERVAL == 0:
                    save_path = f"{save_dir}{step//1000}k.pkl"
                policy_net.save(save_path)

            # if episode is over (no lives left etc), then reset and start new episode
            if done_tplus1:
                rewards_buffer.append(episode_rewards[episode])
                lengths_buffer.append(episode_lengths[episode])

                break

            # phi_t = phi_tplus1

    # * End For
    # * End For


if __name__ == "__main__":
    main()
