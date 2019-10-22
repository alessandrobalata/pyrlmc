from problems.problem import Parameters
import numpy as np
import matplotlib.pyplot as plt


class Performance(Parameters):
    def __init__(self):
        super().__init__()
        self.values = np.zeros((self.N + 1, self.M))

    def compute(self, n, x, u):
        print('computing the one step performance')
        self.values[n + 1, :] = self.values[n, :] + self.running_reward(x, u) * self.dt

    def mean(self, n):
        print(f'computing the average performance at time {n}')
        return np.mean(self.values[n, :])

    def std(self, n):
        print(f'computing the standard error at time {n}')
        return np.std(self.values[n, :]) / self.M

    def plot(self):
        plt.figure()
        plt.plot(self.values)
        plt.xlabel('time step')
        plt.ylabel('performance value')
        plt.title('Performance over time')

    def hist(self):
        plt.figure()
        plt.hist(self.values[self.N, :], bins=max(50, np.sqrt(len(self.values[self.N, :]))))
        plt.xlabel('performance value')
        plt.ylabel('number of occurrences')
        plt.title('Distribution of the estimated value function at time 0')

    def scatter(self, controlled_process, time):
        plt.figure()
        plt.plot(controlled_process.values[time, :], self.values[time + 1, :], 'o')
        plt.xlabel('process value')
        plt.ylabel('performance value')
        plt.title(f'Performance vs. Controlled Process at time {time}')
