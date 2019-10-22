from objects.basis_functions import BasisFunctions
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


class RegressionCoefficients(BasisFunctions):
    '''
    Regression Coefficients, they can be compute() that is an handler of the private method _fit()
    '''

    def __init__(self):
        super().__init__()
        self.values = np.zeros((self.N + 1, self.K)) * np.nan

    def compute(self, n, v, x):
        '''
        handler for the fitting
        :param n: time
        :param v: value function
        :param x: evaluation point(s)
        :return: regression coefficients
        '''
        print('computing the regression coefficients')
        self.values[n, :] = self._fit(v, self.eval(x))[::-1]
        return self.values[n, :].reshape(1, self.K)

    def _fit(self, v, basis):
        '''
        function that fits the vector v on the basis.
        The function needs to transform v into a column vector and the basis needs to be MxK
        :param v: 1xM numpy array
        :param basis: KxM numpy array
        :return: regression coefficients 1xK
        '''
        model = sm.OLS(v.reshape(self.M, 1), basis.T).fit()
        return model.params.reshape(1, self.K)

    def plot(self):
        plt.figure()
        plt.plot(self.values)
        plt.xlabel('time step')
        plt.ylabel('coefficient value')
        plt.legend([f'coeff {x}' for x in range(self.K)])
        plt.title('Regression Coefficients over time')
