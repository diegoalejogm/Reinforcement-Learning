import numpy as np
import math


def zero_value(env):
    # Size of Value Array: (nS, nA)
    size = (env.nS, env.nA)
    return np.zeros(size)


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


def sarsa(env, alpha=0.5, gamma=1, epsilon=.1, num_episodes=200):
    sum_rewards = []
    # Create Q
    Q = zero_value(env)
    # Run for a given number of times
    for t in range(num_episodes):
        sum_rewards.append(0)
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
            sum_rewards[t] += reward
            # Finish episode if done==True
            if done:
                break
    return Q, sum_rewards


def Q_learning(env, alpha=1, gamma=1, epsilon=0.1, num_episodes=1000):
    # Create Q
    Q = zero_value(env)
    # Run for a given number of times
    for _ in range(num_episodes):
        # Obtain initial state
        state = env.reset()
        # Run each episode
        while True:
            # Choose action from env given e-greedy policy given Q
            action = egreedy_policy(state, Q, epsilon)
            # Take action, obtain next state & reward
            next_state, reward, done, _ = env.step(action)
            # Choose next action as max Q(S',a) or equivalently max(Q[s'])
            next_action = np.argmax(Q[next_state])
            # Approximate Q
            Q[state][next_action] += alpha * \
                (reward + gamma * Q[next_state]
                 [next_action] - Q[state, action])
            # Update state variable
            state = next_state
            # Finish episode if done==True
            if done:
                break
    return Q
