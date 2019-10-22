import numpy as np
import matplotlib.pyplot as plt

from problems.problem import Problem


class ControlledProcess:
    '''
    Controlled state process object to be used during forward and backward procedure
    '''

    def __init__(self, problem: Problem):
        self.values = np.zeros((problem.N + 1, problem.M)) * np.nan
        self.values[0, :] = problem.initial_condition
        self.transition_function = problem.transition_function
        self.M = problem.M

    def next_step(self, n: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        Computes the next step of the controlled process
        :param n: time step
        :param x: state vector
        :param u: control vector
        :return: vector of state process at time n+1
        '''
        print('computing the next step of the controlled process')
        self.values[n + 1, :] = self.transition_function(n, x, u)
        return self.values[n + 1, :].reshape(1, self.M)

    def plot(self) -> None:
        '''
        plots the controlled state process vs time
        :return: None
        '''
        plt.figure()
        plt.plot(self.values)
        plt.xlabel('time step')
        plt.ylabel('control process value')
        plt.title('Controlled Process over time')
