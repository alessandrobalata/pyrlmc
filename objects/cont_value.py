from objects.basis_functions import BasisFunctions
import numpy as np

from problems.problem import Problem


class ContValue(BasisFunctions):
    '''
    Continuation Value object, inherits from the basis function object and provides methods to compute the current
    continuation value both backward and forward
    '''

    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.M = problem.M
        self.U = problem.U

    def compute_batch(self, n: int, x: np.ndarray, u: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        '''
        Computes the continuation value using a batch of control values per state, slower than self.evaluate,
        that works on a single control value per state
        :param n: time step
        :param x: space vector
        :param u: control vector (orthogonal (figuratively) to state vector)
        :param coeff: vector of regression coefficients
        :return: matrix of continuation values corresponding to the pair (x[i],u[j])
        '''
        print('computing the continuation value')
        exp = self.compute_exp_batch(n, x, u)
        out = np.zeros((self.M, self.U))
        for j in range(self.U):
            b = exp[:, :, j]
            out[:, j] = np.dot(coeff, b)
        return out

    def evaluate(self, n: int, x: np.ndarray, u: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        '''
        Computes the continuation value by multiplying coefficients with basis functions
        :param n: time step
        :param x: space vector
        :param u: control vector
        :param coeff: vector of regression coefficients
        :return: vector of continuation value
        '''
        print('evaluating the continuation value')
        return np.dot(coeff, self.compute_exp(n, x, u.T))

    def derivative(self, n: int, x: np.ndarray, u: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        '''

        :param n:
        :param x:
        :param u:
        :param coeff:
        :return:
        '''
        return np.dot(coeff, self.compute_der(n, x, u.T))

    def second_derivative(self, n: int, x: np.ndarray, u: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        '''

        :param n:
        :param x:
        :param u:
        :param coeff:
        :return:
        '''
        return np.dot(coeff, self.compute_sec_der(n, x, u.T))
