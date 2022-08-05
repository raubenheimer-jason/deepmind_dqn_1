"""
Changes:
-> Using BreakoutNoFrameskip-v4

# TODO
-> Loss of a life sets "done" flag to true (otherwise no penalty for loosing a life)
-> manual "frame skip" where it takes the max over two frames as well...
-> "states" need to overlap: s1 = {x1, x2, x3, x4}, s2 = {x2, x3, x4, x5} (assuming that x1->x2 has already taken frame skipping into account...)
"""


from collections import deque
import random
import torch
# from game import Preprocessing
import numpy as np
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from env_wrappers import make_env
from helpers import State
from dqn import Network, select_action, calc_loss, init_weights

from constants import *


# TODO use wrapper to do frame skipping so we can accumulate rewards for the skipped frames
# --> https://github.com/PacktPublishing/Hands-on-Reinforcement-Learning-with-PyTorch/blob/master/Section%203/3.7%20Dueling%20DQN%20with%20Pong.ipynb


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

    env = make_env("BreakoutNoFrameskip-v4")

    # # game holds the env
    # # env = gym.make("ALE/Breakout-v5",
    # env = gym.make("BreakoutNoFrameskip-v4",
    #                #    render_mode="human",  # or rgb_array
    #                render_mode="rgb_array",  # or human
    #                new_step_api=True)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayScaleObservation(env)
    # env = gym.wrappers.FrameStack(env, 4, new_step_api=True)
    num_actions = env.action_space.n
    # env_obs_space = env.observation_space

    # set seed
    seed = 31
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # * Initialize replay memory D to capacity N
    replay_mem = deque(maxlen=REPLAY_MEM_SIZE)  # replay_mem is D
    # Need to fill the replay_mem (to REPLAY_START_SIZE) with the results from random actions
    #   -> maybe do this in the main loop and just select random until len(replay_mem) >= REPLAY_START_SIZE

    # # * Initialize action-value function Q with random weights Theta
    # # initialise policy_net
    # # policy_net = Network(num_actions, env_obs_space).to(device)
    # policy_net = Network(num_actions, AGENT_HISTORY_LEN).to(device)
    # policy_net.apply(init_weights)

    # # * Initialize target action-value function Q_hat with weights Theta_bar = Theta
    # # initialise target_net
    # # target_net = Network(num_actions, env_obs_space).to(device)
    # target_net = Network(num_actions, AGENT_HISTORY_LEN).to(device)
    # target_net.load_state_dict(policy_net.state_dict())

    # # # # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    # # optimiser = torch.optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE, alpha=0.99,
    # #                                 eps=1e-08, weight_decay=0, momentum=GRADIENT_MOMENTUM, centered=False, foreach=None)
    # optimiser = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    step = 0
    replay_mem_size = 0
    epoch_count = 0

    episode_rewards = []
    episode_lengths = []

    rewards_buffer = deque([], maxlen=100)
    lengths_buffer = deque([], maxlen=100)
    # loss_buffer = deque([], maxlen=100)

    a_t = 0  # defined here to do the "frame skipping"

    prev_life = 0

    state = State()

    training_episodes = 0

    # * For episode = 1, M do
    for episode in count():

        episode_rewards.append(0.0)
        episode_lengths.append(0)

        # * Initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)
        # phi_t=1, preprocessed sequence
        if prev_life == 0:
            # only reset if lost all lives
            frame, info = env.reset(return_info=True)
            state.add_frame(frame)
            phi_t = state.get_state()
            prev_life = info['lives']

        # define r_t here so we can accumulate rewards over the skipped frames
        r_t = 0.0

        # * For t = 1, T do
        for t in count():
            step += 1

            # * With probability epsilon select a random action a_t
            # * otherwise select a_t = argmax_a Q(phi(s_t),a;Theta)
            if step % ACTION_REPEAT == 0:
                # "frame-skipping" technique where agent only selects a new action on every kth frame.
                # running step requires a lot less computation than having the agent select action
                # this allows roughly k times more games to be played without significantly increasing runtime
                # skipped frames done get added to the memory etc...
                # and rewards are accumulated over the skipped frames (https://stats.stackexchange.com/questions/287670/reinforcement-learning-reward-in-skip-frame)
                # not sure if that is a reliable source...
                a_t = select_action(num_actions, step,
                                    phi_t, policy_net, device)

            # * Execute action a_t in emulator and observe reward r_t and image x_t+1
            # new_frame, r_t, term, trun, info = env.step(a_t)
            new_frame, reward, term, trun, info = env.step(a_t)
            r_t += reward

            # if lost a life then mark the end of an episode so the agent gets a negative result for loosing a life
            if not prev_life == info['lives']:
                lost_life = True
                prev_life = info['lives']
            else:
                lost_life = False

            # done flag (terminated or truncated or lost_life)
            done_tplus1 = term or trun or lost_life

            if step % ACTION_REPEAT == 0:
                # this is the k'th frame which we dont skip...

                # clip reward
                if r_t > REWARD_MAX:
                    r_t = REWARD_MAX
                if r_t < REWARD_MIN:
                    r_t = REWARD_MIN

                # * Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                state.add_frame(new_frame)
                # get new state
                phi_tplus1 = state.get_state()

                episode_rewards[episode] += r_t
                episode_lengths[episode] += 1

                # * Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)

                # * Store transition (phi_t, a_t, r_t, phi_t+1) in D
                # added done flag (tplus1 to matach phi_tplus1)
                transition = (phi_t, a_t, r_t, phi_tplus1, done_tplus1)
                replay_mem.append(transition)  # replay_mem is D
                replay_mem_size += 1

                phi_t = phi_tplus1

                # don't take minibatch until replay mem has been initialised
                if replay_mem_size > REPLAY_START_SIZE:
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

                    for param in policy_net.parameters():  # gradient clipping
                        # https://github.com/jacobaustin123/pytorch-dqn/blob/master/dqn.py
                        # https://www.reddit.com/r/MachineLearning/comments/4dnyiz/question_about_loss_clipping_on_deepminds_dqn/
                        # print(type(param), param.size())
                        param.grad.data.clamp_(-1, 1)

                    optimiser.step()

                    epoch_count += 1

            if step > REPLAY_START_SIZE:
                # * Every C steps reset Q_hat = Q
                # Update Target Network
                if step % TARGET_NET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # Logging
            if (LOGGING or PRINTING) and step % LOG_INTERVAL == 0:
                rew_mean = np.mean(rewards_buffer) or 0
                len_mean = np.mean(lengths_buffer) or 0

                if PRINTING:
                    print()
                    print('Step (total)', step)
                    print('Avg Rew (mean last 100 episodes)', rew_mean)
                    print('Avg Ep steps (mean last 100 episodes)', len_mean)
                    print('(training) Episodes', training_episodes)
                    print('epoch_count', epoch_count)

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

                if replay_mem_size > REPLAY_START_SIZE:
                    training_episodes += 1

                break

    # * End For
    # * End For


if __name__ == "__main__":
    main()
