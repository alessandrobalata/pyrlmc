from objects.basis_functions import BasisFunctions
from objects.cont_value import ContValue
from objects.control import Control
from objects.regression_coefficients import RegressionCoefficients
from objects.training_points import TrainingPoints
from objects.value_function import ValueFunction
from problems.problem import Problem


class BackwardIteration:
    def __init__(self, problem: Problem):
        self.training_points = TrainingPoints(problem)
        self.control = Control(problem)
        self.value_function = ValueFunction(problem)
        self.cont_value = ContValue(problem)
        self.basis = BasisFunctions(problem)
        self.regression_coefficients = RegressionCoefficients(problem)
        self.N = problem.N

    def run(self):
        x = self.training_points.generate(self.N)
        v = self.value_function.terminal_condition(x)
        for n in range(self.N):
            print(f'looping over iteration {n}/{self.N}')
            coeff = self.regression_coefficients.compute(self.N - n, v, x)
            x = self.training_points.generate(self.N - n)
            control = self.control.compute(self.N - n, x, self.cont_value, coeff)
            cont_value = self.cont_value.evaluate(self.N - n, x, control, coeff)
            v = self.value_function.compute(self.N - n, x, control, cont_value)
        return self.regression_coefficients
