
import random
import torch
import torch.nn as nn


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
    )

    # compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(
            observation_space.sample()[None]).float()).shape[1]

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
        super(Network, self).__init__()

        self.num_actions = num_actions
        self.env_obs_space = env_obs_space

        conv_net = nature_cnn(env_obs_space)

        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

    def forward(self, x):
        return self.net(x)

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
