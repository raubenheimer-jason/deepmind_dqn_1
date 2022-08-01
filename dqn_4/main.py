

from collections import deque
import random
import torch
from dqn import Network, select_action, calc_loss, REPLAY_START_SIZE, init_weights
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

REPLAY_MEM_SIZE = int(1e6)
BATCH_SIZE = 32
# LEARNING_RATE = 0.25e-3  # learning rate used by RMSProp
LEARNING_RATE = 5e-5  # learning rate used by RMSProp
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
SAVING = False


def get_frames(p):
    """Get Numpy representation without dumping the frames."""
    return np.concatenate(p, axis=0)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
            # noops = random.randint(
            #     1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, term, trun, _ = self.env.step(self.noop_action)
            done = term or trun
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


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
                   #    render_mode="human",  # or human
                   new_step_api=True)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4, new_step_api=True)
    env = NoopResetEnv(env, noop_max=30)

    #! need to add "max pix value" to observation...
    num_actions = env.action_space.n
    env_obs_space = env.observation_space

    # * Initialize replay memory D to capacity N
    replay_mem = deque(maxlen=REPLAY_MEM_SIZE)  # replay_mem is D
    # Need to fill the replay_mem (to REPLAY_START_SIZE) with the results from random actions
    #   -> maybe do this in the main loop and just select random until len(replay_mem) >= REPLAY_START_SIZE
    print("initialising replay buffer")
    obs = env.reset()
    for step in range(REPLAY_START_SIZE):
        action = env.action_space.sample()
        obs_plus1, rew, term, trun, info = env.step(action)  # x_tplus1
        # print(step, obs_plus1, rew, term, trun)
        done_tplus1 = term or trun  # done flag (terminated or truncated)
        # Store transition (obs, action, rew, obs+1) in D
        # added done flag (tplus1 to matach obs_plus1)
        transition = (obs, action, rew, obs_plus1, done_tplus1)
        replay_mem.append(transition)  # replay_mem is D
        obs = obs_plus1
        if done_tplus1:
            obs = env.reset()
    print("done initialising replay buffer")

    # * Initialize action-value function Q with random weights Theta
    # initialise policy_net
    policy_net = Network(num_actions, env_obs_space).to(device)
    policy_net.apply(init_weights)

    # * Initialize target action-value function Q_hat with weights Theta_bar = Theta
    # initialise target_net
    target_net = Network(num_actions, env_obs_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    # target_net.eval()

    # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    # optimiser = torch.optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE, alpha=0.99,
    #                                 eps=1e-08, weight_decay=0, momentum=GRADIENT_MOMENTUM, centered=False, foreach=None)

    optimiser = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    step = 0

    episode_rewards = []
    episode_lengths = []

    rewards_buffer = deque([], maxlen=100)
    lengths_buffer = deque([], maxlen=100)

    a_t = 0  # defined here to do the "frame skipping"

    # * For episode = 1, M do
    for episode in count():

        episode_rewards.append(0.0)
        episode_lengths.append(0)

        # * Initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)
        # phi_t=1, preprocessed sequence
        # phi_t, _ = env.reset(return_info=True)
        phi_t = env.reset(return_info=True)
        print("reset")

        # * For t = 1, T do
        for t in count():
            step += 1

            # * With probability epsilon select a random action a_t
            # * otherwise select a_t = argmax_a Q(phi(s_t),a;Theta)
            if step % ACTION_REPEAT == 0:
                # "frame-skipping" technique where agent only selects a new action on every kth frame.
                # running step requires a lot less computation than having the agent select action
                # this allows roughly k times more games to be played without significantly increasing runtime
                # act_phi_t = np.stack([get_frames(p) for p in phi_t])
                # act_phi_t = np.stack([get_frames(phi_t)])
                a_t = select_action(num_actions, step,
                                    phi_t, policy_net, device)

            # * Execute action a_t in emulator and observe reward r_t and image x_t+1
            phi_tplus1, r_t, term, trun, info = env.step(a_t)  # x_tplus1

            # print(info["lives"])

            done_tplus1 = term or trun  # done flag (terminated or truncated)

            episode_rewards[episode] += r_t
            episode_lengths[episode] += 1

            # * Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)

            # * Store transition (phi_t, a_t, r_t, phi_t+1) in D
            # added done flag (tplus1 to matach phi_tplus1)
            transition = (phi_t, a_t, r_t, phi_tplus1, done_tplus1)
            replay_mem.append(transition)  # replay_mem is D

            phi_t = phi_tplus1

            # don't take minibatch until replay mem has been initialised
            # if step > REPLAY_START_SIZE:
            if True:
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
                    target_net.load_state_dict(policy_net.state_dict())

            # Logging
            if LOGGING and step % LOG_INTERVAL == 0:
                rew_mean = np.mean(rewards_buffer) or 0
                len_mean = np.mean(lengths_buffer) or 0

                print()
                print('Step', step)
                print('Avg Rew (mean last 100 episodes)', rew_mean)
                print('Avg Ep steps (mean last 100 episodes)', len_mean)
                print('Episodes', episode)

                summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
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
