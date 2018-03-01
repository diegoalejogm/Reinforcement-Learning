import sys
sys.path.append("..")

import numpy as np
import classes as c
import utils as u
import random

def dynaQ(env, step_size, epsilon, num_experience_reps, num_simulated_reps):
    # Init reward list
    rewards = []
    # Init Q
    action_values = c.ActionValue(u.extract_actions(env))
    # Init model
    model = {}
    # Init e-greedy policy
    policy = c.EGreedyPolicy(epsilon, action_values)
    # Current (non-terminal) state
    state = env.reset()
    for _ in range(num_experience_reps):
        action = policy.sample(state)
        # Execute action A, observe reward and next state
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        # Update value
        action_values[state, action] += step_size * ( reward + action_values.max(next_state) - action_values[state, action])
        # Update Model
        model[state, action] = (reward, next_state)
        # Update state
        if done: state = env.reset()
        else: state = next_state
        for _ in range(num_simulated_reps):
            # Get randomly observed state S and action A
            state_sim, action_sim = random.choice(list(model.keys()))
            # Get reward R and next state S' for (S,A)
            reward_sim, next_state_sim = model[state_sim, action_sim]
            # Update value
            action_values[state_sim, action_sim] += step_size * (reward_sim + action_values.max(next_state_sim) - action_values[state_sim, action_sim])

    return action_values

import gym, sys

# %matplotlib inline
sys.path.append('../..')
sys.path.append('..')

env = gym.make('CliffWalking-v0')
Q = dynaQ(
    env=env, step_size=.5, epsilon=.1,
    num_experience_reps=50, num_simulated_reps=0
)