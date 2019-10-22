from objects.controlled_process import ControlledProcess
import numpy as np
import matplotlib.pyplot as plt

from problems.problem import Problem


class Performance:
    '''
    Performance object, to be used to accumulate performance over time during the forward iteration
    '''

    def __init__(self, problem: Problem):
        self.values = np.zeros((problem.N + 1, problem.M))
        self.running_reward = problem.running_reward
        self.dt = problem.dt
        self.M = problem.M
        self.N = problem.N
        self.terminal_condition_fnc = problem.terminal_condition_fnc

    def compute(self, n: int, x: np.ndarray, u: np.ndarray) -> None:
        '''
        Computes the performance measure at the next time step, given the position of the state process
        :param n: time step
        :param x: state vector
        :param u: control vector
        :return: None
        '''
        print('computing the one step performance')
        self.values[n + 1, :] = self.values[n, :] + self.running_reward(x, u) * self.dt

    def last_step(self, x: np.ndarray) -> None:
        '''
        Computes the terminal condition
        :param x: state vector x
        :return: None
        '''
        print('computing terminal condition')
        self.values[self.N, :] = self.values[self.N, :] + self.terminal_condition_fnc(x)

    def mean(self, n: int) -> float:
        '''
        Computes the average performance at time n
        :param n: time step
        :return: mean performance at time n
        '''
        print(f'computing the average performance at time {n}')
        return np.mean(self.values[n, :])

    def std(self, n: int) -> float:
        '''
        Computes the standard deviation of the performance at time n
        :param n: time step
        :return: standard deviation of the performance
        '''
        print(f'computing the standard error at time {n}')
        return np.std(self.values[n, :]) / np.sqrt(self.M)

    def plot(self) -> None:
        '''
        plots the performance over time
        :return: None
        '''
        plt.figure()
        plt.plot(self.values)
        plt.xlabel('time step')
        plt.ylabel('performance value')
        plt.title('Performance over time')

    def hist(self) -> None:
        '''
        plots an histogram of the performance at terminal time
        :return: None
        '''
        plt.figure()
        plt.hist(self.values[self.N, :], bins=max(50, np.sqrt(len(self.values[self.N, :]))))
        plt.xlabel('performance value')
        plt.ylabel('number of occurrences')
        plt.title('Distribution of the estimated value function at time 0')

    def scatter(self, controlled_process: ControlledProcess, time: int) -> None:
        '''
        plots the performance against the value of the controlled state process at given time
        :param controlled_process: controlled process object
        :param time: time step
        :return: None
        '''
        plt.figure()
        plt.plot(controlled_process.values[time, :], self.values[time + 1, :], 'o')
        plt.xlabel('process value')
        plt.ylabel('performance value')
        plt.title(f'Performance vs. Controlled Process at time {time}')
