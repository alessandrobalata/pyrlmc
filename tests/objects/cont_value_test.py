import math

from objects.basis_functions import BasisFunctions
import numpy as np


class ContValue(BasisFunctions):
    def __init__(self):
        super().__init__()

    def compute_batch(self, n, x, u, coeff):
        print('computing the continuation value')
        exp = self.compute_exp_batch(n, x, u)
        out = np.zeros((self.M, self.U))
        for j in range(self.U):
            b = exp[:, :, j]
            out[:, j] = np.dot(coeff, b)
        return out

    def evaluate(self, n, x, u, coeff):
        print('evaluating the continuation value')
        return np.dot(coeff, self.compute_exp(n, x, u.T))
