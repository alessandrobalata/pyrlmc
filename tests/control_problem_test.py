from unittest import TestCase
from tests.paramtest import ProblemTest
from control_problem import ControlProblem


class ControlProblemTest(TestCase):
    def setUp(self):
        problem = ProblemTest()
        self.control_problem = ControlProblem(problem)
        self.N = problem.N

    def test_solve(self):
        _, performance = self.control_problem.solve()
        solution = 1
        self.assertAlmostEqual(performance.mean(self.N), solution)
