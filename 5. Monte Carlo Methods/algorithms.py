import numpy as np


def random_policy(env):
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    return np.ones(S + (A,)) / A


def zero_action_value(env):
    S = tuple([space.n for space in env.observation_space.spaces])
    A = env.action_space.n
    return np.ones(S + (A,)) / A


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
        # print(state)
        # print(policy[state])
        choice = np.random.choice(indices, p=policy[state])
        next_state, reward, done, info = env.step(choice)
        sequence.append((state, choice, reward))
        state = numerize_state(next_state)
        if done:
            break
    return sequence


def mc_onpolicy_firstvisit(env):
    # Initialize Policy and Q
    policy = random_policy(env)
    Q = zero_action_value(env)
    returns = dict()
    # while True:
    for i in range(100000):
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
            # print(key)
            # print(returns[key])
            # print(Q[key])
        # Epsilon-Soft
        # print('----')
        epsilon = 0.15
        for state, _, _ in sequence:
            # print("SEQ", len(sequence))
            max_action = np.argmax(Q[tuple(state)])
            num_actions = Q.shape[-1]
            equal_fraction = epsilon / num_actions
            for action in range(num_actions):
                key = tuple(state) + (action,)
                # print('-------')
                # print(key)
                # print(policy[key])
                policy[key] = equal_fraction
                if action == max_action:
                    # print("ENTRA")
                    policy[key] += 1 - epsilon
                # print(policy[key])
                # print('-------')

    return policy, Q, returns
