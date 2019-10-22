from objects.basis_functions import BasisFunctions
from objects.cont_value import ContValue
from objects.control import Control
from objects.controlled_process import ControlledProcess
from objects.performance import Performance
from objects.regression_coefficients import RegressionCoefficients
from problems.problem import Problem


class ForwardIteration:
    def __init__(self, problem: Problem, regression_coefficients: RegressionCoefficients):
        self.control = Control(problem)
        self.cont_value = ContValue(problem)
        self.basis = BasisFunctions(problem)
        self.regression_coefficients = regression_coefficients
        self.performance = Performance(problem)
        self.process = ControlledProcess(problem)
        self.N = problem.N

    def run(self):
        x = self.process.values[0, :]
        for n in range(1, self.N):
            print(f'looping over iteration {n}/{self.N}')
            coeff = self.regression_coefficients.values[n, :]
            control = self.control.compute(n, x, self.cont_value, coeff)
            self.performance.compute(n, x, control)
            x = self.process.next_step(n, x, control)
        self.performance.last_step(x)
        self.performance.mean(self.N)
        self.performance.std(self.N)
        return self.performance
