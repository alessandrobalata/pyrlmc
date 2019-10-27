import numpy as np


class DefaultFunctions:
    def __init__(self):
        self.T = None
        self.N = None
        self.K_cus = None
        self.K_pol = None
        self.coeff_rr_u = None
        self.coeff_rr_uu = None
        self.coeff_rr_xu = None
        self.coeff_rr_xx = None
        self.coeff_rr_c = None
        self.coeff_rr_x = None
        self.coeff_sigma_x = None
        self.coeff_sigma_u = None
        self.coeff_sigma_c = None
        self.dt = None
        self.coeff_mu_u = None
        self.coeff_mu_x = None
        self.coeff_mu_c = None
        self.M = None
        self.measure_mu = None
        self.K = None

    def init_params(self):
        self.dt = self.T / self.N
        self.K = self.K_pol + self.K_cus

    def generate_training_points(self, x):
        return self.measure_mu[1] * np.random.randn(self.M) + self.measure_mu[0]

    def transition_function_deterministic(self, n, x, u):
        return (self.coeff_mu_c + self.coeff_mu_x * x + self.coeff_mu_u * u) * self.dt

    def transition_function_stochastic(self, n, x, u):
        return np.sqrt(self.dt) * (self.coeff_sigma_c + self.coeff_sigma_x * x + self.coeff_sigma_u * u)

    def transition_function(self, n, x, u):
        return x + self.transition_function_deterministic(n, x, u) + \
               self.transition_function_stochastic(n, x, u) + np.random.randn(self.M)

    def running_reward(self, x, u):
        return self.coeff_rr_c + self.coeff_rr_x * x + self.coeff_rr_u * u + \
               self.coeff_rr_xx * x ** 2 + self.coeff_rr_uu * u ** 2 + \
               self.coeff_rr_xu * x * u + u * 0 + x * 0

    def first_derivative(self, n, x, u):
        return self.coeff_rr_u + 2 * self.coeff_rr_uu * u + self.coeff_rr_xu * x + u * 0

    def second_derivative(self, n, x, u):
        return 2 * self.coeff_rr_uu + u * 0 + x * 0

    def transition_function_deterministic_du(self, n, x, u):
        return self.coeff_mu_u * self.dt + 0 * u + 0 * x

    def transition_function_stochastic_du(self, n, x, u):
        return np.sqrt(self.dt) * self.coeff_sigma_u + 0 * u + 0 * x

    def transition_function_deterministic_duu(self, n, x, u):
        return 0 * u + 0 * x

    def transition_function_stochastic_duu(self, n, x, u):
        return 0 * u + 0 * x
