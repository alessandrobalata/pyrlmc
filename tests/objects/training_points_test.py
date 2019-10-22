import numpy as np

from objects.training_points import TrainingPoints
from tests.paramtest import ParametersTest


class TrainingPointsTest(ParametersTest):
    def setUp(self):
        super().setUp()
        self.training_points = TrainingPoints()
        M = 100_000
        self.training_points.M = M
        self.training_points.values = np.zeros((self.N + 1, M)) * np.nan

    def test_generate(self):
        returns = self.training_points.generate(1)
        self.assertAlmostEqual(np.mean(returns), 0, 1)
        self.assertAlmostEqual(np.std(returns), 1, 1)
        self.assertEqual(np.shape(returns)[0], 1)
        self.assertEqual(np.shape(returns)[1], self.training_points.M)
