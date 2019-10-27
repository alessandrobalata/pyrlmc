import numpy as np
from typing import Tuple, Any
import math

from problems.problem import Problem


class PolynomialBasisFunctions:
    '''
    Collection of polynomial basis functions
    '''

    def __init__(self, problem: Problem):
        self.K_pol = problem.K_pol
        self.dt = problem.dt
        self.sigma = problem.sigma
        self.transition_function_deterministic = problem.transition_function_deterministic
        self.transition_function_stochastic = problem.transition_function_stochastic
        self.transition_function_deterministic_du = problem.transition_function_deterministic_du
        self.transition_function_stochastic_du = problem.transition_function_stochastic_du
        self.transition_function_deterministic_duu = problem.transition_function_deterministic_duu
        self.transition_function_stochastic_duu = problem.transition_function_stochastic_duu
        self.basis_function, self.basis_function_expectation, self.basis_function_exp_der, \
        self.basis_function_exp_sec_der = self.initialise_basis_function()

    def initialise_basis_function(self) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Initialises basis functions
        :return: returns 2 lists with basis functions and their expectation
        '''
        basis_function, basis_function_expectation, basis_function_exp_der, basis_function_exp_sec_der = [], [], [], []
        for k in range(self.K_pol):
            basis_function.append(self._polynomial(k))
            basis_function_expectation.append(self._polynomial_expectation(k))
            basis_function_exp_der.append(self._polynomial_expectation_du(k))
            basis_function_exp_sec_der.append(self._polynomial_expectation_duu(k))
        basis_function = np.array(basis_function)
        basis_function_expectation = np.array(basis_function_expectation)
        basis_function_exp_der = np.array(basis_function_exp_der)
        basis_function_exp_sec_der = np.array(basis_function_exp_sec_der)
        return basis_function, basis_function_expectation, basis_function_exp_der, basis_function_exp_sec_der

    @staticmethod
    def n_choose_k(n, k):
        '''
        arrange n objects in groups of k. How many different groups can you make?
        :param n:
        :param k:
        :return:
        '''
        f = math.factorial
        return f(n) // f(k) // f(n - k)

    @staticmethod
    def __expectation_normal(n):
        '''

        :param n:
        :return:
        '''
        if n == 0:
            return 1
        elif np.mod(n, 2) == 0:
            num = 2 ** (-n / 2) * math.factorial(n)
            den = math.factorial(n / 2)
            return num / den
        else:
            return 0

    @staticmethod
    def _polynomial(degree: int) -> Any:
        '''
        polynomial basis functions, return a polynomial of given degree
        :param degree: degree of the polynomial
        :return: lambda function
        '''
        def __polynomial(x: np.ndarray) -> np.ndarray:
            '''

            :param x:
            :return:
            '''
            return x ** degree
        return __polynomial

    def _polynomial_expectation(self, degree: int) -> Any:
        '''
        initialises the conditional expectation of the basis functions
        :param degree: int degree of the basis function
        :return: function
        '''
        def __polynomial_expectation(n: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
            '''
            conditional expectation of x_np1 given x_n and u_n
            :param n: time step
            :param x: state vector
            :param u: control vector
            :return: vector of expected values
            '''
            if degree > 0:
                fd = self.transition_function_deterministic(n, x, u)
                fs = self.transition_function_stochastic(n, x, u)
                tmp = 0 * x + 0 * u
                if degree == 1:
                    tmp = x + fd
                else:
                    for d in range(degree + 1):
                        c = self.__expectation_normal(degree - d)
                        tmp += (self.n_choose_k(degree, d) * (x + fd) ** d *
                                c * fs ** (degree - d))
                return tmp
            else:
                return 0 * x + 0 * u + 1

        return __polynomial_expectation

    def _polynomial_expectation_du(self, degree: int) -> Any:
        '''

        :param degree:
        :return:
        '''
        def __polynomial_expectation_du(n: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
            if degree > 0:
                fd = self.transition_function_deterministic(n, x, u)
                fd_u = self.transition_function_deterministic_du(n, x, u)
                fs = self.transition_function_stochastic(n, x, u)
                fs_u = self.transition_function_stochastic_du(n, x, u)
                tmp = 0 * x + 0 * u
                for d in range(degree + 1):
                    c = self.__expectation_normal(degree - d)
                    tmp += c * (d * self.n_choose_k(degree, d) *
                                (x + fd) ** max(0, d - 1) *
                                fd_u *
                                fs ** (degree - d) *
                                +
                                (degree - d) * self.n_choose_k(degree, d) *
                                (x + fd) ** d *
                                fs ** max(0, degree - d - 1) *
                                fs_u)
                return tmp
            else:
                return 0 * x + 0 * u

        return __polynomial_expectation_du

    def _polynomial_expectation_duu(self, degree: int) -> Any:
        '''

        :param degree:
        :return:
        '''
        def __polynomial_expectation_duu(n: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
            if degree > 1:
                fd = self.transition_function_deterministic(n, x, u)
                fd_u = self.transition_function_deterministic_du(n, x, u)
                fd_uu = self.transition_function_deterministic_duu(n, x, u)
                fs = self.transition_function_stochastic(n, x, u)
                fs_u = self.transition_function_stochastic_du(n, x, u)
                fs_uu = self.transition_function_stochastic_duu(n, x, u)
                tmp = 0 * x + 0 * u
                for d in range(degree + 1):
                    c = self.__expectation_normal(degree - d)
                    tmp += c * (d * max(0, d - 1) * self.n_choose_k(degree, d) *
                                (x + fd) ** max(0, d - 2) *
                                fd_u ** 2 *
                                fs ** (degree - d)
                                +
                                d * self.n_choose_k(degree, d) *
                                (x + fd) ** max(0, d - 1) *
                                fd_uu *
                                fs ** (degree - d)
                                +
                                d * (degree - d) * self.n_choose_k(degree, d) *
                                (x + fd) ** max(0, d - 1) *
                                fd_u *
                                fs ** max(0, degree - d - 1)
                                +
                                (degree - d) * d * self.n_choose_k(degree, d) *
                                (x + fd) ** max(0, d - 1) *
                                fd_u *
                                fs ** max(0, degree - d - 1) *
                                fs_u
                                +
                                (degree - d) * max(0, degree - d - 1) * self.n_choose_k(degree, d) *
                                (x + fd) ** d *
                                fs ** max(0, degree - d - 2) *
                                fs_u ** 2
                                +
                                (degree - d) * self.n_choose_k(degree, d) *
                                (x + fd) ** d *
                                fs ** max(0, degree - d - 1) *
                                fs_uu)
                return tmp
            else:
                return 0 * x + 0 * u

        return __polynomial_expectation_duu
