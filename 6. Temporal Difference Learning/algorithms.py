import sys
sys.path.append("..")

import numpy as np
import classes as c
import utils as u


def sarsa(env, alpha=0.5, gamma=1, epsilon=.1, num_episodes=200):
    """
    Returns the Q-value estimates for an environment by using the SARSA
    algorithm (State-Action-Reward-State-Action).

    Parameters
    ----------
    env : gym.core.Env
        OpenAI Gym Environment instance
    alpha : float
        Algorithm's learning rate
    gamma : float
        Discount for next rewards
    epsilon : float
        Probability of choosing an action randomly
    num_episodes : int
        Number of episodes for the policy iteration process

    Returns
    -------
    numpy.ndarray
        Estimated Q (state-action) values
    list
        List of rewards of each episode

    """
    # Stats tracking
    sum_rewards = []
    # Create Q
    Q = c.ActionValue(u.extract_actions(env))
    policy = c.EGreedyPolicy(epsilon, Q)
    # Run for a given number of times
    for t in range(num_episodes):
        sum_rewards.append(0)
        # Obtain initial state
        state = env.reset()
        # Choose action from env given e-greedy policy given Q
        action = policy.sample(state)
        # Run each episode
        while True:
            # Take action, obtain next state & reward
            next_state, reward, done, _ = env.step(action)
            # Choose next action
            next_action = policy.sample(next_state)
            # Approximate Q
            Q[state,action] += alpha * \
                (reward + gamma * Q[next_state, next_action] - Q[state, action])
            # Update state variables
            state = next_state
            action = next_action
            sum_rewards[t] += reward
            # Finish episode if done==True
            if done:
                break
    return Q, sum_rewards


def qlearning(env, alpha=0.5, gamma=1, epsilon=0.1, num_episodes=100):
    """
    Returns the Q-value estimates for an environment by using the Q-Learning
    algorithm.

    Parameters
    ----------
    env : gym.core.Env
        OpenAI Gym Environment instance
    alpha : float
        Algorithm's learning rate
    gamma : float
        Discount for next rewards
    epsilon : float
        Probability of choosing an action randomly
    num_episodes : int
        Number of episodes for the policy iteration process

    Returns
    -------
    numpy.ndarray
        Estimated Q (state-action) values
    list
        List of rewards of each episode

    """
    # Stats tracking
    sum_rewards = []
    # Create Q
    Q = c.ActionValue(u.extract_actions(env))
    policy = c.EGreedyPolicy(epsilon, Q)
    # Run for a given number of times
    for t in range(num_episodes):
        sum_rewards.append(0)
        # Obtain initial state
        state = env.reset()
        # Run each episode
        while True:
            # Choose action from env given e-greedy policy given Q
            action = policy.sample(state)
            # Take action, obtain next state & reward
            next_state, reward, done, _ = env.step(action)
            # Choose next action as max Q(S',a) or equivalently max(Q[s'])
            next_action = Q.argmax(next_state)
            # Approximate Q
            Q[state,action] += alpha * (reward + gamma * Q[next_state,next_action] - Q[state,action])
            # Update state variable
            state = next_state
            sum_rewards[t] += reward
            # Finish episode if done==True
            if done:
                break
    return Q, sum_rewards


def double_qlearning(env, alpha=0.5, gamma=1, epsilon=0.1, num_episodes=100):
    """
    Returns the Q-value estimates for an environment by using the Double
    Q-Learning algorithm.

    Parameters
    ----------
    env : gym.core.Env
        OpenAI Gym Environment instance
    alpha : float
        Algorithm's learning rate
    gamma : float
        Discount for next rewards
    epsilon : float
        Probability of choosing an action randomly
    num_episodes : int
        Number of episodes for the policy iteration process

    Returns
    -------
    numpy.ndarray
        Estimated Q (state-action) values
    list
        List of rewards of each episode

    """
    # Stats tracking
    sum_rewards = []
    # Create Q1 and Q2
    Q1 = c.ActionValue(u.extract_actions(env))
    Q2 = c.ActionValue(u.extract_actions(env))
    # Run for a given number of times
    for t in range(num_episodes):
        sum_rewards.append(0)
        # Obtain initial state
        state = env.reset()
        # Run each episode
        while True:
            # Choose action from env given e-greedy policy given Q1 and Q2
            policy = c.EGreedyPolicy(epsilon, Q1 + Q2)
            action = policy.sample(state)
            # Take action, obtain next state & reward
            next_state, reward, done, _ = env.step(action)
            # Choose policy to update randomly
            if np.random.rand() < .5:
                # Choose next action as max Q(S',a) or equivalently max(Q[s'])
                next_action = Q1.argmax(next_state)
                # Approximate Q
                Q1[state, action] += alpha * \
                    (reward + gamma * Q2[next_state, next_action] - Q1[state, action])
            else:
                # Choose next action as max Q(S',a) or equivalently max(Q[s'])
                next_action = Q2.argmax(next_state)
                # Approximate Q
                Q2[state, action] += alpha * \
                    (reward + gamma * Q1[next_state, next_action] - Q2[state, action])
            # Update state variable
            state = next_state
            sum_rewards[t] += reward
            # Finish episode if done==True
            if done:
                break
    return (Q1 + Q2) / 2, sum_rewards
