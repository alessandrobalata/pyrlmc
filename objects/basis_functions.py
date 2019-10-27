import numpy as np
from objects.misc.polynomial_basis import PolynomialBasisFunctions
from problems.problem import Problem


class BasisFunctions:
    '''
    Basis function objects with methods to compute the conditional expectation
    '''

    def __init__(self, problem: Problem):
        self.K = problem.K
        self.M = problem.M
        self.U = problem.U
        self.pbf = PolynomialBasisFunctions(problem)
        self.basis_function = np.append(self.pbf.basis_function, problem.custom_basis)
        self.basis_function_expectation = np.append(self.pbf.basis_function_expectation, problem.custom_basis_expectation)
        self.basis_function_exp_der = np.append(self.pbf.basis_function_exp_der, problem.custom_basis_exp_der)
        self.basis_function_exp_sec_der = np.append(self.pbf.basis_function_exp_sec_der, problem.custom_basis_exp_der)

    def eval(self, x: np.ndarray) -> np.ndarray:
        '''
        Evaluates the basis functions and returns a matrix with a column per basis function
        :param x: state vector
        :return: matrix MxK
        '''
        out = np.zeros((self.K, len(x.T)))
        for k in range(self.K):
            out[k, :] = self.basis_function[k](x)
        return out

    def compute_exp_batch(self, n: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        Computes expectations for a batch of control values per state value
        :param n: time step
        :param x: state vector
        :param u: control vector
        :return: cube of conditional expectation of basis functions KxMxU
        '''
        out = np.zeros((self.K, self.M, self.U))
        for k in range(self.K):
            out[k, :, :] = self.basis_function_expectation[k](n, x, u.T).T
        return out

    def compute_exp(self, n: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        Computes expectations for a control value per state value
        :param n: time step
        :param x: state vector
        :param u: control vector
        :return: cube of conditional expectation of basis functions KxM
        '''
        out = np.zeros((self.K, self.M))
        for k in range(self.K):
            out[k, :] = self.basis_function_expectation[k](n, x, u.T)
        return out

    def compute_der(self, n: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''

        :param n:
        :param x:
        :param u:
        :return:
        '''
        out = np.zeros((self.K, self.M))
        for k in range(self.K):
            out[k, :] = self.basis_function_exp_der[k](n, x, u.T)
        return out

    def compute_sec_der(self, n: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''

        :param n:
        :param x:
        :param u:
        :return:
        '''
        out = np.zeros((self.K, self.M))
        for k in range(self.K):
            out[k, :] = self.basis_function_exp_sec_der[k](n, x, u.T)
        return out
