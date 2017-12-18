import numpy as np


def zero_value(env):
    # S: Tuple of Num. States
    S = tuple([space.n for space in env.observation_space.spaces])
    # A: Num. Actions
    A = env.action_space.n
    # Size of Value Array: (S0, S1, ... Sn, A)
    size = S + (A,)
    return np.zeros(size) / A


def egreedy_policy(s, Q, epsilon):
    # Obtain random number in range [0,1)
    random = np.random.rand()
    # If random in epsilon, choose random action
    if random < epsilon:
        num_actions = Q[s].shape[0]
        indices = np.arange(num_actions)
        return np.random.choice(indices)
    # Otherwise return greedy action
    return np.argmax(Q[s])


def sarsa(env, alpha, gamma=1, num_episodes=100000):
    # Create Q
    Q = zero_value(env)
    # Run for a given number of times
    for t in num_episodes:
        # Calculate epsilon
        epsilon = 1 / (t + 1)
        # Obtain initial state
        state = env.reset()
        # Choose action from env given e-greedy policy given Q
        action = egreedy_policy(state, Q, epsilon)
        # Run each episode
        while True:
            # Take action, obtain next state & reward
            next_state, reward, done, _ = env.step(action)
            # Choose next action
            next_action = egreedy_policy(next_state, Q, epsilon)
            # Approximate Q
            Q[state][action] += alpha * \
                (reward + gamma * Q[next_state]
                 [next_action] - Q[state, action])
            # Update state variables
            state = next_state
            action = next_action
            # Finish episode if done==True
            if done:
                break
    return Q


def Q_learning(env, alpha, gamma, epsilon=0.1, num_episodes=100000):
    # Create Q
    Q = zero_value(env)
    # Run for a given number of times
    for t in num_episodes:
        # Obtain initial state
        state = env.reset()
        # Choose action from env given e-greedy policy given Q
        action = egreedy_policy(state, Q, epsilon)
        # Run each episode
        while True:
            # Take action, obtain next state & reward
            next_state, reward, done, _ = env.step(action)
            # Choose next action as max Q(S',a) or equivalently max(Q[s'])
            next_action = np.max(Q[next_state])
            # Approximate Q
            Q[state][action] += alpha * \
                (reward + gamma * Q[next_state]
                 [next_action] - Q[state, action])
            # Update state variable
            state = next_state
            # Finish episode if done==True
            if done:
                break
    return Q
