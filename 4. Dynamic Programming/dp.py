import numpy as np

def generate_random_policy(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.ones([n_states, n_actions]) / n_actions
    return policy

def policy_evaluation(policy, env, V=None, gamma=0.9, theta=0.001):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rewards_low, rewards_high = env.reward_range[0], env.reward_range[1]
    # Init V
    if not V: V = np.zeros(n_states)
    # Evaluate policy iteratively
    while True:
        delta = 0
        for s in range(n_states):
            val = V[s]
            V[s] = sum(policy[s,a] * state_transition_prob[s1, r, s, a] * (r + gamma * V[s1]) for a in range(n_actions) for r in range(rewards_low, rewards_high) for s1 in range(n_states))
            delta = max(delta, abs(val - V[s]))
        if theta > delta: break
    return V

def policy_improvement(V, policy, env, gamma=0.9):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rewards_low, rewards_high = env.reward_range[0], env.reward_range[1]

    policy_stable = False
    for s in states:
        wasnt_greedy = np.nonzero(policy[s]).shape[0] > 1
        old_action = np.max(policy[s])
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = sum(state_transition_prob[s1,r,s,a] * (r + gamma * V[s]) for r in range(rewards_low, rewards_high) for s1 in range(n_states))
        best_action = np.argmax(action_values)
        V[s] = action_values[best_action]
        policy[s, :] = 0
        policy[s, besbest_actiont_a] = 1
        policy_changed = old_action is not best_action
        if wasnt_greedy or policy_changed : policy_stable = False
    return policy_stable, policy

def policy_iteration(policy, env):
    V = None
    while True:
        V = policy_evaluation(V)
        policy_stable, policy = policy_improvement()
        if policy_stable: break
    return V, policy
