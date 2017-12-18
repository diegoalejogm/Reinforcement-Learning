import numpy as np


def random_policy(env):
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    return np.ones(S + (A,)) / A


def deterministic_policy(env):
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    policy = np.zeros(S + (A,)) / A
    policy[..., 0] = 1
    return policy


def zero_action_value(env):
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    return np.zeros(S + (A,)) / A


def zero_weight_cumulative_sum(env):
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    return np.zeros(S + (A,)) / A


def numerize_state(state):
    numerized = 1 if state[2] else 0
    return (state[0], state[1], numerized)


def generate_sequence(policy, env):
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    sequence = []

    state = numerize_state(env.reset())
    indices = np.arange(A)
    while True:
        # print(policy[state])
        choice = np.random.choice(indices, p=policy[state])
        next_state, reward, done, info = env.step(choice)
        sequence.append((state, choice, reward))
        state = numerize_state(next_state)
        if done:
            break
    return sequence


def mc_onpolicy_firstvisit(env, num_simulations=50000):
    # Initialize Policy and Q
    policy = random_policy(env)
    Q = zero_action_value(env)
    returns = dict()
    # while True:
    for _ in range(num_simulations):
        # Generate sequence given Policy
        sequence = generate_sequence(policy, env)
        # First Visit
        G = 0
        for state, action, reward in reversed(sequence):
            G += reward
            key = tuple(state) + (action,)
            if key not in returns:
                returns[key] = []
            # Update Returns and Q based on new return
            returns[key].append(G)
            Q[key] = sum(returns[key]) / float(len(returns[key]))
        # Epsilon-Soft
        epsilon = 0.15
        for state, _, _ in sequence:
            max_action = np.argmax(Q[tuple(state)])
            num_actions = Q.shape[-1]
            equal_fraction = epsilon / num_actions
            for action in range(num_actions):
                key = tuple(state) + (action,)
                policy[key] = equal_fraction
                if action == max_action:
                    policy[key] += 1 - epsilon

    return policy, Q, returns


def mc_offpolicy(env, num_simulations=1000000, gamma=1):
    # Initialize Policy and values
    policy = deterministic_policy(env)
    b = random_policy(env)
    Q = zero_action_value(env)
    C = zero_weight_cumulative_sum(env)

    for _ in range(num_simulations):
        sequence = generate_sequence(b, env)
        G, W = 0.0, 1.0
        for state, action, reward in reversed(sequence):
            #            print(state, action, reward)
            G += (gamma * G) + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * \
                (G - Q[state][action])
            argmax_action = np.argmax(Q[state])
            policy[state] = 0
            policy[state][argmax_action] = 1
            # Early break if non-greedy action taken
            if action != np.argmax(policy[state]):
                break
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
