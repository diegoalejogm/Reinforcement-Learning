import numpy as np
import abc

class Agent(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, environment):
        self.action_space = environment.action_space
        self.Q = [ 0 for i in range (self.action_space.shape) ]

    @abc.abstractmethod
    def act(self):
        pass

    @abc.abstractmethod
    def observe(self, reward):
        pass

class RandomAgent(Agent):

    def __init__(self, environment):
        Agent.__init__(self, environment)

    def act(self):
        return self.action_space.sample()

    def observe(self, reward):
        pass
