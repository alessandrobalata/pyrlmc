from objects.cont_value import ContValue
from objects.control import Control
import numpy as np

from tests.paramtest import ParametersTest


class ControlTest(ParametersTest):
    def setUp(self):
        super().setUp()
        self.control = Control()
        self.cont_value = ContValue()

    def test_compute(self):
        n = 1
        M = 1000
        x = np.array([1] * M).reshape(1, M)
        coeff = np.array([0, 1, 0])
        returns = self.control.compute(n=n, x=x, cont_value=self.cont_value, coeff=coeff)
        self.assertAlmostEqual(returns[0][0], -0.45226131, 3)
        self.assertEqual(np.shape(returns)[1], M)
