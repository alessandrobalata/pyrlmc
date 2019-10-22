from typing import Union
import numpy as np

from objects.basis_functions import BasisFunctions
from tests.objects.polynomial_basis_test import PolynomialBasisFunctionsTest


class BasisFunctionsTest(PolynomialBasisFunctionsTest):
    def setUp(self):
        super().setUp()
        self.basis_fun = BasisFunctions()

    def test_eval(self):
        M = 1000
        x = np.array([1] * M)
        returns = self.basis_fun.eval(x=x)
        self.assertEqual(np.shape(returns)[1], M)
        self.assertEqual(np.shape(returns)[0], 3)

    # TODO test and improve the following 2 methods
    def compute_exp_batch(self, n: float, x: Union[float, np.ndarray], u: Union[float, np.ndarray]):
        out = np.zeros((self.K, self.M, self.U))
        for k in range(self.K):
            # TODO remove this if, should not be necessary
            if k == 0:
                out[k, :, :] = self.basis_function_expectation[k](x, u.T)
            else:
                out[k, :, :] = self.basis_function_expectation[k][0](x, u.T).T
        return out

    def compute_exp(self, n: float, x: Union[float, np.ndarray], u: Union[float, np.ndarray]):
        out = np.zeros((self.K, self.M))
        for k in range(self.K):
            # TODO remove this if, should not be necessary
            if k == 0:
                out[k, :] = self.basis_function_expectation[k](x, u.T)
            else:
                out[k, :] = self.basis_function_expectation[k][0](x, u.T)
        return out
