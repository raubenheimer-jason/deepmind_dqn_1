

from collections import deque
import random
import time
import torch
from dqn import Network
# from game import Preprocessing
from game import Game, Preprocessing, game_test
import numpy as np
import gym
from itertools import count

from agent import Agent

REPLAY_MEM_SIZE = int(1e6)
BATCH_SIZE = 32
GAMMA = 0.99  # discount factor used in Q-learning update
LEARNING_RATE = 0.25e-3  # learning rate used by RMSProp
GRADIENT_MOMENTUM = 0.95  # RMSProp
SQUARED_GRADIENT_MOMENTUM = 0.95  # RMSProp
MIN_SQUARED_GRADIENT = 0.01  # RMSProp
TARGET_NET_UPDATE_FREQ = int(1e4)  # C

INITIAL_EXPLORATION = 1  # Initial value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION = 0.1  # final value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION_FRAME = int(1e6)  # num frames epsilon changes linearly

REPLAY_START_SIZE = int(5e4)  # uniform random policy run before learning

PRINT_INFO_FREQ = int(1e3)


# DECAY_SLOPE = (1.0-0.1)/(0.0-1e6)
# delta y over delta x
DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
    (REPLAY_START_SIZE-(REPLAY_START_SIZE+FINAL_EXPLORATION_FRAME))
DECAY_C = INITIAL_EXPLORATION - (DECAY_SLOPE*REPLAY_START_SIZE)
# DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
#     (0.0-FINAL_EXPLORATION_FRAME)


def main():
    # start = time.time()
    # game_test()

    # print(time.time()-start)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # return

    # game holds the env
    env = gym.make("ALE/Breakout-v5",
                   render_mode="rgb_array",  # or human
                   new_step_api=True)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4, new_step_api=True)
    #! need to add "max pix value" to observation...
    num_actions = env.action_space.n
    env_obs_space = env.observation_space

    # agent = Agent()

    # preprocess = Preprocessing()

    # * Initialize replay memory D to capacity N
    replay_mem = deque(maxlen=REPLAY_MEM_SIZE)  # replay_mem is D
    # Need to fill the replay_mem (to REPLAY_START_SIZE) with the results from random actions
    #   -> maybe do this in the main loop and just select random until len(replay_mem) >= REPLAY_START_SIZE

    # get number of actions
    # https://stackoverflow.com/questions/63113154/how-to-check-out-actions-available-in-openai-gym-environment
    # num_actions = env.action_space.n
    # num_actions = game.num_actions

    # * Initialize action-value function Q with random weights Theta
    # initialise policy_net
    policy_net = Network(num_actions, env_obs_space).to(device)

    # * Initialize target action-value function Q_hat with weights Theta_bar = Theta
    # initialise target_net
    target_net = Network(num_actions, env_obs_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    optimiser = torch.optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE, alpha=0.99,
                                    eps=1e-08, weight_decay=0, momentum=GRADIENT_MOMENTUM, centered=False, foreach=None)

    step = 0

    # * For episode = 1, M do
    for episode in count():

        # * Initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)
        # phi_t=1, preprocessed sequence
        phi_t, _ = env.reset(return_info=True)

        # * For t = 1, T do
        for t in count():
            step += 1

            if step % PRINT_INFO_FREQ == 0:
                print(f"step: {step}, t: {t}, episode: {episode}")

            # * With probability epsilon select a random action a_t
            # * otherwise select a_t = argmax_a Q(phi(s_t),a;Theta)
            a_t = select_action(num_actions, step, phi_t, policy_net)

            # * Execute action a_t in emulator and observe reward r_t and image x_t+1
            phi_tplus1, r_t, term, trun, info = env.step(a_t)  # x_tplus1
            done_tplus1 = term or trun  # done flag (terminated or truncated)

            # * Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)
            # s_tplus1 = [s_t, a_t, x_tplus1]
            # # s.append(a_t)
            # # s.append(x_tplus1)  # creating s_t+1
            # # calculate preprocessed phi at t+1 using s_t+1
            # phi_tplus1 = preprocess.process(s_tplus1)  # phi_t+1

            # * Store transition (phi_t, a_t, r_t, phi_t+1) in D
            # added done flag (tplus1 to matach phi_tplus1)
            transition = (phi_t, a_t, r_t, phi_tplus1, done_tplus1)
            replay_mem.append(transition)  # replay_mem is D

            # don't take minibatch until replay mem has been initialised
            if step > REPLAY_START_SIZE:
                # * Sample random minibatch of transitions (phi_j, a_j, r_j, phi_j+1) from D
                minibatch = random.sample(replay_mem, BATCH_SIZE)

                # * Set y_j = r_j if episode terminates at step j+1
                # * otherwise set y_j = r_j + gamma * max_a_prime Q_hat(phi_j+1, a_prime; Theta_bar)
                # * Perform a gradient descent step on (y_j - Q(phi_j, a_j; Theta))^2 with respect to the network parameters Theta

                # calculate loss [ (y_j - Q(phi_j, a_j; Theta))^2 ]
                loss = calc_loss(minibatch, target_net, policy_net)

                # Gradient Descent
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # * Every C steps reset Q_hat = Q
                # Update Target Network
                if step % TARGET_NET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # if episode is over (no lives left etc), then reset and start new episode
            if done_tplus1:
                break

            phi_t = phi_tplus1

    # * End For
    # * End For


def select_action(num_actions, step, phi_t, policy_net):
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

    else:
        # step must be <= 5e4, still in initialise replay mem state
        # setting epsilon = 1 ensures that we always choose a random action
        # random.random --> the interval [0, 1), which means greater than or equal to 0 and less than 1
        epsilon = 1

    rand_sample = random.random()

    if step >= REPLAY_START_SIZE:
        print(f"epsilon: {epsilon}, rand_sample: {rand_sample}")

    if rand_sample < epsilon:
        action = random.randrange(num_actions)
        # return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
        # print(f"random action: {action}")
    else:
        with torch.no_grad():
            policy_q = policy_net(phi_t)
            # max_q_indices = torch.argmax(policy_q, dim=1)
            max_q_index = torch.argmax(policy_q, dim=1)
            # actions = max_q_indices.detach().tolist()
            action = max_q_index.detach().tolist()
            # return self.policy_net(state).max(1)[1].view(1, 1)
            # print(f"policy_q action: {action}")

    return action


def calc_loss(minibatch, target_net, policy_net):
    """ calculates loss: (y_j - Q(phi_j, a_j; theta))^2

        calculating targets y_j:
        y_j = r_j if episode terminates at step j+1
        otherwise
        y_j = r_j + gamma * "max_target_q_values"

        minibatch = batch of transitions (phi_t, a_t, r_t, phi_tplus1, done)

    """

    #! could probably do this as "tensor operations" rather and a for loop...

    y_j = []  # targets (uses Q_hat)

    policy_q = []  # policy network (uses Q)

    a_j = []

    for transition in minibatch:
        r_j = transition[2]

        if transition[4] == True:
            # done == true
            y_j.append(r_j)

        else:
            phi_jplus1 = transition[3]
            target_q_values = target_net(phi_jplus1)
            max_target_q = target_q_values.max(dim=1, keepdim=True)[0]
            y_j_val = r_j + GAMMA * max_target_q
            y_j.append(y_j_val)

        phi_j = transition[0]
        policy_q.append(policy_net(phi_j))

        a_j.append(transition[1])

    # now calc loss
    action_q_values = torch.gather(input=policy_q, dim=1, index=a_j)
    loss = torch.nn.functional.smooth_l1_loss(action_q_values, y_j)

    return loss


if __name__ == "__main__":
    # print("Agent")
    # init_game()

    main()
