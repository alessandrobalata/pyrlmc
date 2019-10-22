from objects.cont_value import ContValue
from problems.problem import Parameters
import numpy as np


class ValueFunction(Parameters):
    def __init__(self):
        super().__init__()
        self.values = np.zeros((self.N+1, self.M)) * np.nan

    def terminal_condition(self, x):
        print('computing terminal condition')
        self.values[self.N, :] = self.terminal_condition_fnc(x)
        return self.values[self.N, :].reshape(1, self.M)

    def compute(self, n, x, u, cont_value):
        print('computing the value function')
        self.values[n, :] = self.running_reward(x, u) * self.dt + cont_value
        return self.values[n, :].reshape(1,self.M)