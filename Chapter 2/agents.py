import numpy as np
import abc

class ValueAgent(object):

    def __init__(self, environment, decision_making_policy, action_value_estimation_method=None, initial_values=None, store_rewards=False, alpha=0):
        self.action_space = environment.action_space

        n = int ( self.action_space.n )
        self.Q_init = np.zeros(n) if not initial_values else initial_values
        self.Q = np.copy(self.Q_init)
        self.N = np.zeros(n)
        self.R = [ [] for i in range(n) ] if store_rewards else None
        self.t = 0
        self.alpha = alpha

        self.avm = action_value_estimation_method
        self.dmp = decision_making_policy

        self.last_action = None

    def act(self, update_state=True):
        action = self.dmp.select_action(self.Q, N=self.N, t=self.t)

        if update_state:
            self.last_action = action
            self.N[self.last_action] += 1
            self.t += 1

        return action

    def observe(self, reward):
        if self.R is not None:
            self.R[self.last_action].append(reward)
        if self.avm: self.avm(self.Q, self.last_action, reward, N=self.N, t=self.t, Q_init=self.Q_init, R=self.R, alpha=self.alpha)
