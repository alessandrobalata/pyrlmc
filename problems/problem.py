import numpy as np

from objects.misc.default_functions import DefaultFunctions


class Problem(DefaultFunctions):
    '''
    Parametes object. Effectively represents each problem that we want to submit to the solver.
    Refer to objects/default_functions.py for the meaning of each parameter.
    '''

    def __init__(self):
        super().__init__()
        # solver parameters
        self.N = 100
        self.M = 1000
        self.K_pol = 3
        self.K_cus = 0
        self.U = 200
        self.u_min, self.u_max = -10, 10
        self.optimization_type = 'extensive'  # 'gradient'
        self.coefficients_computation =  'ols'

        # control problem
        self.T = 1

        # controlled process
        self.initial_condition = 0
        self.sigma = 2
        self.coeff_mu_c = 0
        self.coeff_mu_x = 0
        self.coeff_mu_u = 1
        self.coeff_sigma_c = 0
        self.coeff_sigma_x = 0
        self.coeff_sigma_u = 1

        # reward function
        self.coeff_rr_c = 0
        self.coeff_rr_x = 0
        self.coeff_rr_xx = 0
        self.coeff_rr_u = 0
        self.coeff_rr_uu = 1
        self.coeff_rr_xu = 0

        # training points
        self.measure_mu = 0, 1

        # terminal condition
        self.terminal_condition_fnc = lambda x: x ** 2

        # gradient descent optimization
        self.step_gradient = 1
        self.epsilon_gradient = 0.0001

        # bayesian regression
        self.epsilon_variance = 0.01
        self.smoothness = 0.1
        self.variance_growth = 0.1
        self.max_trust = 0.1

        # initialization - ignore
        self.custom_basis = np.array([])
        self.custom_basis_expectation = np.array([])
        self.custom_basis_exp_der = np.array([])
        self.init_params()
