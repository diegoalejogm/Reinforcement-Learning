import numpy as np


def generate_random_policy(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.ones([n_states, n_actions]) / n_actions
    policy[0, :] = 0
    policy[n_states - 1, :] = 0
    return policy


def policy_evaluation(policy, env, V=None, gamma=1, theta=1e-8, max_t=None):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rewards_low, rewards_high = env.reward_range[0], env.reward_range[1]
    # Init V
    if V is None:
        V = np.zeros(n_states)
    # Evaluate policy iteratively
    t = 0
    while True:
        t += 1
        delta = 0
        for s in range(1, n_states - 1):
            val = V[s]
            V[s] = sum(policy[s, a] * env.state_transition_prob(s1, r, s, a) * (r + gamma * V[s1])
                       for a in range(n_actions) for r in range(rewards_low, rewards_high + 1) for s1 in range(n_states))
            delta = max(delta, abs(val - V[s]))
        if theta > delta or (max_t and t == max_t):
            break
    return V


def policy_improvement(V, policy, env, gamma=1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rewards_low, rewards_high = env.reward_range[0], env.reward_range[1]

    policy_stable = True
    for s in range(1, n_states - 1):
        old_action = np.argmax(policy[s])
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = sum(env.state_transition_prob(s1, r, s, a) * (r + gamma * V[s1])
                                   for r in range(rewards_low, rewards_high + 1) for s1 in range(n_states))
        best_action = np.argmax(action_values)
        # Update Policy
        policy[s, :] = 0
        policy[s, best_action] = 1
        policy_changed = old_action != best_action
        if policy_changed:
            policy_stable = False
    return policy, policy_stable


def print_policy(policy):
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    for s in range(1, policy.shape[0] - 1):
        print('State:{} Action: {}'.format(s, actions[np.argmax(policy[s])]))


def policy_iteration(policy, env):
    V = None
    while True:
        V = policy_evaluation(policy, env, V=V)
        policy, policy_stable = policy_improvement(V, policy, env)
        if policy_stable:
            break
    return V, policy


def value_iteration(policy, env, gamma=1, theta=1e-8):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rewards_low, rewards_high = env.reward_range[0], env.reward_range[1]
    # Init V
    V = np.zeros(n_states)
    # Evaluate policy iteratively
    while True:
        delta = 0
        for s in range(1, n_states - 1):
            val = V[s]
            V[s] = max(env.state_transition_prob(env.next_state(s, a), r, s, a) * (r + gamma * V[env.next_state(s, a)])
                       for a in range(n_actions) for r in range(rewards_low, rewards_high + 1))

            delta = max(delta, abs(val - V[s]))
        if theta >= delta:
            break
    # Output deterministic policy
    for s in range(1, n_states - 1):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = sum(env.state_transition_prob(s1, r, s, a) * (r + gamma * V[s1])
                                   for r in range(rewards_low, rewards_high + 1) for s1 in range(n_states))
        print(action_values)
        best_action = np.argmax(action_values)
        # Update Policy
        policy[s, :] = 0
        policy[s, best_action] = 1
    return V, policy
