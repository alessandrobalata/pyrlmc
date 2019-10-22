from objects.cont_value import ContValue
from objects.controlled_process import ControlledProcess
import numpy as np
import matplotlib.pyplot as plt

from problems.problem import Problem


class Control:
    '''
    Control object to be used both backward and forward
    '''

    def __init__(self, problem: Problem):
        self.values = np.zeros((problem.N + 1, problem.M)) * np.nan
        self.u_max = problem.u_max
        self.u_min = problem.u_min
        self.U = problem.U
        self.running_reward = problem.running_reward
        self.dt = problem.dt
        self.M = problem.M
        self.optimization_type = problem.optimization_type
        self.step_gradient = problem.step_gradient
        self.epsilon_gradient = problem.epsilon_gradient
        self.first_derivative = problem.first_derivative
        self.second_derivative = problem.second_derivative
        self.N = problem.N

    def compute(self, n: int, x: np.ndarray, cont_value: ContValue, coeff: np.ndarray) -> np.ndarray:
        '''

        :param n:
        :param x:
        :param cont_value:
        :param coeff:
        :return:
        '''
        if self.optimization_type == 'extensive':
            return self._extensive_search(n, x, cont_value, coeff)
        elif self.optimization_type == 'gradient' and n < self.N - 3:
            u_tp1 = x * 0
            return self._gradient_descent(n, x, u_tp1, cont_value, coeff)
        return self._extensive_search(n, x, cont_value, coeff)

    def _gradient_descent(self, n: int, x: np.ndarray, u_tp1: np.ndarray, cont_value: ContValue, coeff: np.ndarray) -> \
            np.ndarray:
        '''

        :param x:
        :param coeff:
        :param u_tp1:
        :return:
        '''
        u = u_tp1
        convergence = False
        while not convergence:
            tmp = u - self.__ratio_derivatives(n, x, u, cont_value, coeff) * self.step_gradient
            variation = np.abs(tmp - u) / (1 + np.abs(u))
            convergence = variation.all() < self.epsilon_gradient
            u = tmp
        self.values[n, :] = u
        return self.values[n, :].reshape(1, self.M)

    def __ratio_derivatives(self, n: int, x: np.ndarray, u: np.ndarray, cont_value: ContValue, coeff: np.ndarray) -> \
            np.ndarray:
        '''

        :param x:
        :param u:
        :param coeff:
        :return:
        '''
        numerator = cont_value.derivative(n, x, u, coeff) + self.first_derivative(n, x, u) * self.dt
        denumerator = cont_value.second_derivative(n, x, u, coeff) + self.second_derivative(n, x, u) * self.dt
        return numerator / denumerator

    def _extensive_search(self, n: int, x: np.ndarray, cont_value: ContValue, coeff: np.ndarray) -> np.ndarray:
        '''
        Computes the optimal control by testing a number of control values in the interval u_min, u_max
        :param n: time step
        :param x: state vector
        :param cont_value: continuation value object
        :param coeff: vector of regression coefficients
        :return: control vector
        '''
        print('computing the control')
        test = np.linspace(self.u_min, self.u_max, self.U).reshape(1, self.U)
        idx = np.argmin(self.running_reward(x, test.T).T * self.dt +
                        cont_value.compute_batch(n, x.reshape(1, self.M), test, coeff),
                        axis=1)
        self.values[n, :] = test[0, idx]
        return self.values[n, :].reshape(1, self.M)

    def plot(self) -> None:
        '''
        Plots the control process over time
        :return: None
        '''
        plt.figure()
        plt.plot(self.values)
        plt.xlabel('time step')
        plt.ylabel('control value')
        plt.title('Control Process over time')

    def scatter(self, controlled_process: ControlledProcess, time: int) -> None:
        '''
        Plots the control process against the values of the controlled process at a given time
        :return: None
        '''
        plt.figure()
        plt.plot(controlled_process.values[time, :], self.values[time, :], 'o')
        plt.xlabel('process value')
        plt.ylabel('control value')
        plt.title('Control vs. Controlled Process')
