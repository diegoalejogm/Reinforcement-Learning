import numpy as np


def random_policy(env):
    """
    Returns a numpy ndarray with size: ( |S0|, |S1|, ..., |Sn|, |A| ) where
    every state has an uniform distribution for selecting every action.
    """
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    return np.ones(S + (A,)) / A


def deterministic_policy(env):
    """
    Returns a numpy ndarray with size: ( |S0|, |S1|, ..., |Sn|, |A| ) where
    every state has a deterministic policy.
    """
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    policy = np.zeros(S + (A,)) / A
    # Give first action probability of 1
    policy[..., 0] = 1
    return policy


def zero_action_value(env):
    """
    Returns a zeroed numpy ndarray with size: ( |S0|, |S1|, ..., |Sn|, |A| ).
    """
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    return np.zeros(S + (A,)) / A


def zero_weight_cumulative_sum(env):
    """
    Returns a zeroed numpy ndarray with size: ( |S0|, |S1|, ..., |Sn|, |A| ).
    """
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    return np.zeros(S + (A,)) / A


def numerize_state(state):
    """
    Given a state tuple (int, int, bool), return the respective tuple in the
    form (int, int, int) where the bool value is mapped to the range [0, 1].
    """
    numerized = 1 if state[2] else 0
    return (state[0], state[1], numerized)


def generate_sequence(policy, env):
    """
    Given a policy and an environment, generate a sequence of states and actions
    which reach to a final state.
    """
    # Init S, A and other variables
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    indices = np.arange(A)
    sequence = []

    # Obtain start state
    state = numerize_state(env.reset())

    while True:
        # Choose an action randomly, using probability distribution policy[s]
        choice = np.random.choice(indices, p=policy[state])
        # Execute action
        next_state, reward, done, info = env.step(choice)
        # Update sequence with new action
        sequence.append((state, choice, reward))
        # Set current state as next_state
        state = numerize_state(next_state)
        # Stop when next_state is final
        if done:
            break
    return sequence


def mc_onpolicy_firstvisit(env, num_simulations=50000):
    """
    Run the Monte Carlo First Visit On-Policy algorithm and return the estimated
    policy, Q (state action) values, and returns (rewards) dict.

    Parameters
    ----------
    num_simulations : int
        Number of episodes for the policy iteration process

    Returns
    -------
    numpy.ndarray
        Estimated Policy
    numpy.ndarray
        Estimated Q (state-action) values
    dict
        Rewards obtained for every state

    """
    # Initialize Policy and Q
    policy = random_policy(env)
    Q = zero_action_value(env)
    returns = dict()
    for _ in range(num_simulations):
        # Generate sequence given Policy
        sequence = generate_sequence(policy, env)
        # Init rewards
        G = 0
        # Traverse sequence backwards
        for state, action, reward in reversed(sequence):
            G += reward
            # State-action tuple
            tupl = tuple(state) + (action,)
            # Add new list for non-existing state-action tuple
            if tupl not in returns:
                returns[tupl] = []
            # Update Returns and Q based on new return
            returns[tupl].append(G)
            Q[tupl] = sum(returns[tupl]) / float(len(returns[tupl]))
        # Epsilon-Soft
        epsilon = 0.15
        for state, _, _ in sequence:
            max_action = np.argmax(Q[tuple(state)])
            num_actions = Q.shape[-1]
            # Calculate probability for subobtimal actions
            equal_fraction = epsilon / num_actions
            for action in range(num_actions):
                # State-action tuple
                tupl = tuple(state) + (action,)
                # Set all actions probability of epsilon / |A|
                policy[tupl] = equal_fraction
                # Add (1 - epsilon) probabilty to optimal action
                if action == max_action:
                    policy[key] += 1 - epsilon

    return policy, Q, returns


def mc_offpolicy(env, gamma=1., num_simulations=1000000):
    """
    Run the Monte Carlo First Visit Off-Policy algorithm with
    Weighted Importance-sampling, a deterministic target policy and a random
    behavior policy.

    Parameters
    ----------
    gamma : float
        Next states reward discount rate
    num_simulations : int
        Number of episodes for the policy iteration process

    Returns
    -------
    numpy.ndarray
        Estimate Policy
    numpy.ndarray
        Estimated Q (state-action) values

    """
    # Initialize policies and values
    policy = deterministic_policy(env)
    b = random_policy(env)
    Q = zero_action_value(env)
    # Denominator
    C = zero_weight_cumulative_sum(env)

    for _ in range(num_simulations):
        # Generate sequence
        sequence = generate_sequence(b, env)
        # Init reward sum and current weight
        G, W = 0.0, 1.0
        # Traverse sequence backwards
        for state, action, reward in reversed(sequence):
            # Update G, C and Q
            G += (gamma * G) + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * \
                (G - Q[state][action])
            argmax_action = np.argmax(Q[state])
            # Set probability suboptimal states to 0
            policy[state] = 0
            # Set probability for optimal state to 1
            policy[state][argmax_action] = 1
            # Early break if non-greedy action taken
            if action != np.argmax(policy[state]):
                break
            # Update W
            W /= b[state][action]

    return policy, Q


def print_deterministic_policy(policy, Q):
    # Usable Ace
    for dealer in range(1, 11):
        for hand in range(11, 22):
            print(policy[hand, dealer, 0])
            action = 'HIT' if np.argmax(
                policy[hand, dealer, 0]) == 1 else 'STICK'
            print(
                "Dealer Showing: {}, Player Sum: {} --> {}, Q:{}".format(dealer, hand, action, Q[hand, dealer, 0]))
