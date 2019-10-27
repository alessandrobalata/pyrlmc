from objects.basis_functions import BasisFunctions
import statsmodels.api as sm
import matplotlib.pyplot as plt
from problems.problem import *


class RegressionCoefficients(BasisFunctions):
    '''
    Regression Coefficients, they can be compute() that is an handler of the private method _fit()
    '''

    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.values = np.zeros((problem.N + 1, problem.K)) * np.nan
        # self.coefficients_computation = problem.coefficients_computation

    def compute(self, n: int, v: np.ndarray, x: np.ndarray) -> np.ndarray:
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

    def _fit(self, v: np.ndarray, basis: np.ndarray) -> np.ndarray:
        '''
        function that fits the vector v on the basis.
        The function needs to transform v into a column vector and the basis needs to be MxK
        :param v: 1xM numpy array
        :param basis: KxM numpy array
        :return: regression coefficients 1xK
        '''
        if True:#self.coefficients_computation == 'ols':
            model = sm.OLS(v.reshape(self.M, 1), basis.T).fit()
            return model.params.reshape(1, self.K)
        # elif self.coefficients_computation == 'bayesian':
        #     pass

    def plot(self) -> None:
        '''
        Plots the Regression coefficients over time
        :return: None
        '''
        plt.figure()
        plt.plot(self.values)
        plt.xlabel('time step')
        plt.ylabel('coefficient value')
        plt.legend([f'coeff {x}' for x in range(self.K)])
        plt.title('Regression Coefficients over time')
