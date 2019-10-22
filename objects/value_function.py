import numpy as np

from problems.problem import Problem


class ValueFunction:
    '''
    Value function object to be used during backward iteration
    '''

    def __init__(self, problem: Problem):
        self.values = np.zeros((problem.N + 1, problem.M)) * np.nan
        self.N = problem.N
        self.terminal_condition_fnc = problem.terminal_condition_fnc
        self.M = problem.M
        self.running_reward = problem.running_reward
        self.dt = problem.dt

    def terminal_condition(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes the terminal condition
        :param x: state vector x
        :return: value function at terminal time N
        '''
        print('computing terminal condition')
        self.values[self.N, :] = self.terminal_condition_fnc(x)
        return self.values[self.N, :].reshape(1, self.M)

    def compute(self, n: int, x: np.ndarray, u: np.ndarray, cont_value: np.ndarray) -> np.ndarray:
        '''
        Computes the value function backwards by summing the current reward to the continuation value
        :param n: time step
        :param x: state vector
        :param u: control vector
        :param cont_value: continuation value
        :return: value function at time n and state x
        '''
        print('computing the value function')
        self.values[n, :] = self.running_reward(x, u) * self.dt + cont_value
        return self.values[n, :].reshape(1, self.M)
