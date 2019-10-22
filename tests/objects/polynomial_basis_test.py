import numpy as np
from typing import Tuple, Any

from tests.paramtest import ParametersTest


class PolynomialBasisFunctionsTest(ParametersTest):
    '''
    Collection of polynomial basis functions
    '''

    def __init__(self):
        super().__init__()

        phi_0_hat = lambda x, u: np.ones(len(x))
        phi_1_hat = lambda x, u: x + u * self.dt,
        phi_2_hat = lambda x, u: x ** 2 + u ** 2 * self.dt ** 2 + 2 * x * u * self.dt + self.dt * self.sigma ** 2,
        phi_3_hat = lambda x, u: x ** 3,
        phi_4_hat = lambda x, u: x ** 4,

        self.basis_function, self.basis_function_expectation = self.initialise_basis_function(
            (phi_0_hat, phi_1_hat, phi_2_hat, phi_3_hat, phi_4_hat))

    def initialise_basis_function(self, exp_basis_list: tuple) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Initialises basis functions
        :param exp_basis_list: list of expectations of basis function
        :return: returns 2 lists with basis functions and their expectation
        '''
        basis_function, basis_function_expectation = [], []
        for k in range(self.K):
            basis_function.append(self.polynomial(k))
            basis_function_expectation.append(exp_basis_list[k])
        basis_function = np.array(self.basis_function)
        basis_function_expectation = np.array(self.basis_function_expectation)
        return basis_function, basis_function_expectation

    @staticmethod
    def polynomial(degree: int) -> Any:
        '''
        polynomial basis functions, return a polynomial of given degree
        :param degree: degree of the polynomial
        :return: lambda function
        '''
        return lambda x: x ** degree

    # def polynomial_expectation(self, n: int, degree: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    #     return self.transition_function_deterministic(n, x, u) ** degree
