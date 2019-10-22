import numpy as np


class Problem:
    '''
    Parametes object. Effectively represents each problem that we want to submit to the solver
    '''

    def __init__(self):
        self.N = 100
        self.M = 1000
        self.K_pol = 3
        self.K_cus = 0
        self.U = 200
        self.T = 1
        self.u_min, self.u_max = -10, 10
        self.initial_condition = 0
        self.sigma = 2

        self.transition_function_deterministic = lambda n, x, u: x + u * self.dt
        self.transition_function_stochastic = lambda n, x, u: np.sqrt(self.dt) * self.sigma * np.random.randn(self.M)
        self.terminal_condition_fnc = lambda x: x ** 2
        self.running_reward = lambda x, u: x ** 2 + u ** 2

        self.transition_function = lambda n, x, u: self.transition_function_deterministic(n, x, u) + \
                                                   self.transition_function_stochastic(n, x, u)
        self.generate_training_points = lambda x: np.random.randn(self.M)
        self.dt = self.T / self.N
        self.K = self.K_pol + self.K_cus

        self.custom_basis = np.array([])
        self.custom_basis_expecation = np.array([])
