import numpy as np

from problems.problem import Problem


class TrainingPoints:
    '''
    Training point object
    '''

    def __init__(self, problem: Problem):
        self.values = np.zeros((problem.N + 1, problem.M)) * np.nan
        self.generate_training_points = problem.generate_training_points
        self.M = problem.M

    def generate(self, n: int) -> np.ndarray:
        '''
        Generates training points at time n
        :param n: time step
        :return: vector of training points
        '''
        print('generating training points')
        self.values[n, :] = self.generate_training_points(n)
        return self.values[n, :].reshape(1, self.M)
