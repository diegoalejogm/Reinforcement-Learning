import sys

sys.path.append('../..')
sys.path.append('..')

import classes as c
import utils as u


def n_step_sarsa(num_steps, env, num_episodes, policy=None, step_size=.5, discount_rate=.9, epsilon=.3):

    episode_rewards = []

    # Init Q(s,a) arbitrarily, for all s in S, a in A(s)
    action_values = c.ActionValue(u.extract_actions(env))
    # Init PI to be e-greedy w.r.t. Q, or to a fixed given policy
    if policy is None:
        policy = c.EGreedyPolicy(epsilon)

    # Store and access operation lists
    states  = c.NstepMemory(num_steps)
    actions = c.NstepMemory(num_steps)
    rewards = c.NstepMemory(num_steps)

    for ep in range(num_episodes):
        # Init and store S0
        states[0], rewards[0] = env.reset(), 0
        # Select and store action A0
        actions[0] = policy.sample(states[0], action_values)
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
                    actions[step+1] = policy.sample(states[step+1], action_values)

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