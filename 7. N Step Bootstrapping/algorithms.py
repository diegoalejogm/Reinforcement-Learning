import sys, random

sys.path.append('../..')
sys.path.append('..')

import classes as c
import utils as u


def n_step_sarsa(num_steps, env, num_episodes, policy=None, step_size=.5, discount_rate=.9, epsilon=.3):
    """
    Returns the Q-value estimates for an environment by using the On-Policy N-Step SARSA algorithm

    Parameters
    ----------
    num_steps: int
        Number of steps used in value function bootstrapping
    env : gym.core.Env
        OpenAI Gym Environment instance
    num_episodes : int
        Number of episodes to be run using the environment
    policy : Policy instance
        Optional Behaviour Policy for which the Q-values will be estimated
        If None, an e-greedy policy will be used, based on the estimated Q-Values
    step_size: float
        Algorithm's learning rate
    discount_rate : float
        Discount rate used when estimating values with future rewards
    epsilon : float
        Probability of choosing an action randomly in e-greedy policy created if
        None 'policy' parameter is used

    Returns
    -------
    ActionValue
        Estimated Q (state-action) values
    list
        List of rewards of each episode

    """
    episode_rewards = []

    # Init Q(s,a) arbitrarily, for all s in S, a in A(s)
    action_values = c.ActionValue(u.extract_actions(env))
    # Init PI to be e-greedy w.r.t. Q, or to a fixed given policy
    if policy is None:
        policy = c.EGreedyPolicy(epsilon, action_values)

    # Store and access operation lists
    states  = c.NstepMemory(num_steps)
    actions = c.NstepMemory(num_steps)
    rewards = c.NstepMemory(num_steps)

    for ep in range(num_episodes):
        # Init and store S0
        states[0], rewards[0] = env.reset(), 0
        # Select and store action A0
        actions[0] = policy.sample(states[0])
        # T: terminal_step = infinity
        step, terminal_step = 0, float('inf')

        episode_rewards.append(0)

        while True:
            if step < terminal_step:
                # Take action a_t. Observe and store Rt+1, St+1
                states[step+1], rewards[step+1], done, _ = env.step(actions[step])
                episode_rewards[ep] += rewards[step+1]
                if done:
                    terminal_step = step + 1
                else:
                    actions[step+1] = policy.sample(states[step+1])

            update_step = step - num_steps + 1

            if update_step >= 0:
                # Calculate n-step return: G
                G = .0
                for i in range(update_step + 1, min(update_step + num_steps+1, terminal_step)):
                    G += discount_rate ** (i - update_step - 1) * rewards[i]
                if update_step + num_steps < terminal_step:
                    G += discount_rate ** (num_steps) * action_values[states[update_step + num_steps], actions[update_step + num_steps]]

                action_values[states[update_step], actions[update_step]] += step_size * (
                    G - action_values[states[update_step], actions[update_step]])

            if update_step == terminal_step - 1:
                break
            else:
                step += 1

    return action_values, episode_rewards


def n_step_backup_tree(num_steps, env, num_episodes, policy=None, step_size=.5, discount_rate=.9, epsilon=.3):
    """
    Returns the Q-value estimates for an environment by using the Off-Policy N-Step Backup Tree Algorithm

    Parameters
    ----------
    num_steps: int
        Number of steps used in value function bootstrapping
    env : gym.core.Env
        OpenAI Gym Environment instance
    num_episodes : int
        Number of episodes to be run using the environment
    policy : Policy instance
        Optional Target Policy for which the Q-values will be estimated
        If None, an e-greedy policy will be used, based on the estimated Q-Values
    step_size: float
        Algorithm's learning rate
    discount_rate : float
        Discount rate used when estimating values with future rewards
    epsilon : float
        Probability of choosing an action randomly in e-greedy policy created if
        None 'policy' parameter is used

    Returns
    -------
    ActionValue
        Estimated Q (state-action) values
    list
        List of rewards of each episode

    """
    episode_rewards = []

    # Init Q(s,a) arbitrarily, for all s in S, a in A(s)
    action_values = c.ActionValue(u.extract_actions(env))
    # Init PI to be e-greedy w.r.t. Q, or to a fixed given policy
    if policy is None:
        policy = c.EGreedyPolicy(epsilon, action_values)

    # Store and access operation lists
    states  = c.NstepMemory(num_steps)
    actions = c.NstepMemory(num_steps)
    old_values = c.NstepMemory(num_steps)
    td_errors = c.NstepMemory(num_steps)
    taken_policy = c.NstepMemory(num_steps)

    for ep in range(num_episodes):
        u.display_episode_log(ep+1, num_episodes)
        # Init default values
        old_values[0], taken_policy[0] = 0, 0
        # Init and store S0
        states[0] = env.reset()
        # Select and store action A0
        actions[0] = policy.sample(states[0])
        # Store Q_t-1(S0, A0) as Q0
        old_values[0] = action_values[states[0], actions[0]]
        # T: terminal_step = infinity
        step, terminal_step = 0, float('inf')
        episode_rewards.append(0)

        while True:
            if step < terminal_step:
                # Take action a_t. Observe and store Rt+1, St+1
                states[step+1], reward, done, _ = env.step(actions[step])
                episode_rewards[ep] += reward
                if done:
                    terminal_step = step + 1
                    td_errors[step] = reward - old_values[step]
                else:
                    expected_values = [policy[states[step + 1], action] * action_values[states[step + 1], action] for action in
                                       action_values.actions]
                    td_errors[step] = reward - old_values[step] + discount_rate * \
                                       sum( expected_values )
                    # Select arbitrarily and store an action as A[t+1]
                    actions[step+1] = random.choice(action_values.actions)
                    # Store Q_t[ A[t+1] | S[t+1] ] as Q_t
                    old_values[step+1] = action_values[states[step+1], actions[step+1]]
                    # Store pi[A[t+1] | S[t+1]] as pi_t
                    taken_policy[step+1] = policy[states[step], actions[step]]


            update_step = step - num_steps + 1

            if update_step >= 0:
                Z = 1.
                G = old_values[update_step]

                for k in range(update_step, min(update_step+num_steps, terminal_step)):
                    G += Z * td_errors[k]
                    if k != (min(update_step+num_steps, terminal_step) - 1):
                        Z *= discount_rate * taken_policy[k + 1]
                action_values[states[update_step], actions[update_step]] += step_size*(G - action_values[states[update_step], actions[update_step]])
            if update_step == terminal_step - 1:
                break
            else:
                step += 1

    return action_values, episode_rewards


def n_step_sarsa_importance_sampling(behaviour_policy, num_steps, env, num_episodes, policy=None, step_size=.5, discount_rate=.9, epsilon=.3):
    """
    Returns the Q-value estimates for an environment by using the Off-Policy N-Step SARSA algorithm

    Parameters
    ----------
    behaviour_policy : Policy instance
        Behaviour Policy for which the Q-values will be estimated
    num_steps: int
        Number of steps used in value function bootstrapping
    env : gym.core.Env
        OpenAI Gym Environment instance
    num_episodes : int
        Number of episodes to be run using the environment
    policy : Policy instance
        Optional Target Policy for which the Q-values will be estimated
        If None, an e-greedy policy will be used, based on the estimated Q-Values
    step_size: float
        Algorithm's learning rate
    discount_rate : float
        Discount rate used when estimating values with future rewards
    epsilon : float
        Probability of choosing an action randomly in e-greedy policy created if
        None 'policy' parameter is used

    Returns
    -------
    ActionValue
        Estimated Q (state-action) values
    list
        List of rewards of each episode

    """

    episode_rewards = []

    # Init Q(s,a) arbitrarily, for all s in S, a in A(s)
    action_values = c.ActionValue(u.extract_actions(env))
    # Init PI to be e-greedy w.r.t. Q, or to a fixed given policy
    if policy is None:
        policy = c.EGreedyPolicy(epsilon, action_values)

    # Store and access operation lists
    states  = c.NstepMemory(num_steps)
    actions = c.NstepMemory(num_steps)
    rewards = c.NstepMemory(num_steps)

    for ep in range(num_episodes):
        # Init and store S0
        states[0], rewards[0] = env.reset(), 0
        # Select and store action A0
        actions[0] = behaviour_policy.sample(states[0])
        # T: terminal_step = infinity
        step, terminal_step = 0, float('inf')

        episode_rewards.append(0)

        while True:
            if step < terminal_step:
                # Take action a[t]. Observe and store R[t+1], S[t+1]
                states[step+1], rewards[step+1], done, _ = env.step(actions[step])
                episode_rewards[ep] += rewards[step+1]
                if done:
                    terminal_step = step + 1
                else:
                    actions[step+1] = behaviour_policy.sample(states[step+1])

            update_step = step - num_steps + 1

            if update_step >= 0:
                # Calculate n-step return: G
                G = .0
                importance_weight = 1
                for i in range(update_step+1, min(update_step + num_steps, terminal_step)):
                    importance_weight *= policy[states[i], actions[i]] / behaviour_policy[states[i], actions[i]]
                for i in range(update_step+1, min(update_step+num_steps+1, terminal_step+1)):
                    G += discount_rate ** (i - update_step - 1) * rewards[i]
                if update_step + num_steps < terminal_step:
                    G += discount_rate ** (num_steps) * action_values[states[update_step + num_steps], actions[update_step + num_steps]]

                action_values[states[update_step], actions[update_step]] += (step_size * importance_weight) * (
                    G - action_values[states[update_step], actions[update_step]])

            if update_step == terminal_step - 1:
                break
            else:
                step += 1

    return action_values, episode_rewards