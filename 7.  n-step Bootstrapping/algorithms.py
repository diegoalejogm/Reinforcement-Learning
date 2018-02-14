import math
import numpy as np

def extract_actions(env):
    return list(range(env.nA))

class NstepMemory:

    def __init__(self, n):
        self.n = n+1
        self.list = []

    def __getitem__(self, step):
        item_index = self.__getIndex__(step)
        return self.list[item_index]

    def __getIndex__(self, step):
        return step % self.n

    def __setitem__(self, step, value):
        value_index = self.__getIndex__(step)
        # Append if list size is not big enough.
        if len(self.list) == value_index:
            self.list.append(value)
        # Replace if list size is bigger.
        elif len(self.list) > value_index:
            self.list[value_index] = value
        # Throw exception if list size is too small.
        else:
            raise IndexError('Current step is skipping some previous step.')


class ActionValue:

    def __init__(self, actions):
        self.dict = dict()
        self.actions = actions
        self.default_value = 0

    def __getitem__(self, item):
        # Raise error less or more than two keys were passed.
        if len(item) != 2:
            raise ValueError('Expected two keys')

        # Get item
        state, action = item
        if (state, action) in self.dict:
            return self.dict[state, action]
        else:
            return self.default_value

    def __setitem__(self, key, value):
        # Raise error less or more than two keys were passed.
        if len(key) != 2:
            raise ValueError('Expected two keys')

        # Set item
        self.dict[key] = value

    def argmax(self, state):
        max = -float('inf')
        argmax = None
        for a in self.actions:
            current_value = self[state, a]
            if current_value > max:
                max = current_value
                argmax = a
        return argmax


class EGreedyPolicy:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def sample(self, state, action_values):
        # Obtain random number in range [0,1)
        random = np.random.rand()
        # If random in epsilon, choose random action
        if random < self.epsilon:
            rand_index = np.random.randint(0, len(action_values.actions))
            return action_values.actions[rand_index]

        # Otherwise return greedy action
        return action_values.argmax(state)


def n_step_sarsa(num_steps, env, num_episodes, policy=None, step_size=.5, discount_rate=.9, epsilon=.3):

    episode_rewards = []

    # Init Q(s,a) arbitrarily, for all s in S, a in A(s)
    action_values = ActionValue(extract_actions(env))
    # Init PI to be e-greedy w.r.t. Q, or to a fixed given policy
    if policy is None:
        policy = EGreedyPolicy(epsilon)

    # Store and access operation lists
    # states, actions, rewards = dict(), dict(), dict()
    states, actions, rewards = NstepMemory(
        num_steps), NstepMemory(num_steps), NstepMemory(num_steps)

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