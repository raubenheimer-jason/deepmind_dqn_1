
from game import init_game

if __name__ == "__main__":
    print("Agent")
    init_game()

    # Initialize replay memory D to capacity N

    # Initialize action-value function Q with random weights Theta

    # Initialize target action-value function Q_hat with weights Theta_bar = Theta

    # Loop
    """
    for episode = 1, M do
        Initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)

        for t = 1, T do
            With probability epsilon select a random action a_t
            otherwise select a_t = argmax_a Q(phi(s_t),a;Theta)

            Execute action a_t in emulator and observe reward r_t and image x_t+1

            Set s_t+1 = s_t,a_t,x_t+1 and preprocess phi_t+1 = phi(s_t+1)

            Store transition (phi_t, a_t, r_t, phi_t+1) in D

            Sample random minibatch of transitions (phi_j, a_j, r_j, phi_j+1) from D

            Set y_j = r_j if episode terminates at step j+1
            otherwise set y_j = r_j + gamma * max_a_prime Q_hat(phi_j+1, a_prime; Theta_bar)

            Perform a gradient descent step on (y_j - Q(phi_j, a_j; Theta))^2 with respect to the network parameters Theta

            Every C steps reset Q_hat = Q


    """
