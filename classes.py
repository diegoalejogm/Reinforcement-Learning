import numpy as np
import copy

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

    def __add__(self, other):

        # Create a copy to return
        clone = copy.deepcopy(self)
        # Merge unique actions and replace
        clone.actions = list(set(clone.actions + other.actions))
        clone += other
        return clone

    def __iadd__(self, other):

        # For every key in the other action_value
        for k in other.dict:
            # Sum if key also exists in cloned
            if k in self.dict:
                self.dict[k] += other.dict[k]
            # Set otherwise as new value for key
            else:
                self.dict[k] = other.dict[k]
        return self

    def __truediv__(self, other):
        # Create a copy to return
        clone = copy.deepcopy(self)

        for k in clone.dict:
            clone.dict[k] /= other

        return clone

    def __div__(self, other):
        return self.__truediv__(other)


    def argmax(self, state):
        max = -float('inf')
        argmax = None
        for a in self.actions:
            current_value = self[state, a]
            if current_value > max:
                max = current_value
                argmax = a
        return argmax



#---- Policies
class EGreedyPolicy:

    def __init__(self, epsilon, action_values):
        self.epsilon = epsilon
        self.action_values = action_values


    def __getitem__(self, item):
        # Raise error less or more than two keys were passed.
        if len(item) != 2:
            raise ValueError('Expected two keys')

        # Get item
        state, action = item
        return self.probability(state, action)

    def probability(self, state, action):
        # Num actions
        n_actions = len(self.action_values.actions)
        # Get argmax item

        argmax = self.action_values.argmax(state)
        if argmax == None:
            raise ValueError("Could't find greedy action: action values returned None.")
        # Calculate probabilty
        probability = self.epsilon / float(n_actions)
        if action == argmax:
            probability += 1.-self.epsilon
        return probability


    def sample(self, state):
        # Obtain random number in range [0,1)
        random = np.random.rand()
        # If random in epsilon, choose random action
        if random < self.epsilon:
            rand_index = np.random.randint(0, len(self.action_values.actions))
            return self.action_values.actions[rand_index]

        # Otherwise return greedy action
        return self.action_values.argmax(state)

