from algorithms.backward_iteration import BackwardIteration
from algorithms.forward_iteration import ForwardIteration
from objects.performance import Performance
from problems.problem import Problem


class ControlProblem:
    def __init__(self, problem):
        super().__init__()
        self.backward_procedure = BackwardIteration(problem)
        self.forward_procedure = None
        self.problem = problem

    def solve(self):
        print('start backward computation of the solution...')
        regression_coefficients = self.backward_procedure.run()
        print('\nstart forward evaluation of the solution...')
        self.forward_procedure = ForwardIteration(self.problem, regression_coefficients)
        performance: Performance = self.forward_procedure.run()
        final = performance.mean(self.problem.N)
        std = performance.std(self.problem.N)
        print(f'\nEstimation for the value function is {final}, with 95% CI [{final - 2 * std}, {final + 2 * std}]')
        return regression_coefficients, performance

    def plot_results(self):
        self.backward_procedure.regression_coefficients.plot()
        self.forward_procedure.process.plot()
        self.forward_procedure.control.plot()
        self.forward_procedure.control.scatter(controlled_process=self.forward_procedure.process, time=50)
        self.forward_procedure.performance.plot()
        self.forward_procedure.performance.hist()
        self.forward_procedure.performance.scatter(controlled_process=self.forward_procedure.process, time=10)


if __name__ == '__main__':
    prob = Problem()
    control_problem = ControlProblem(prob)
    regression_coeff, perf = control_problem.solve()
    control_problem.plot_results()
