from objects.controlled_process import ControlledProcess
from tests.paramtest import ParametersTest
import numpy as np


class ControlledProcessTest(ParametersTest):
    def setUp(self):
        super().setUp()
        self.controlled_process = ControlledProcess()

    def test_next_step(self):
        M = 1000
        x_t = np.array([1]*M).reshape(1, M)
        u = np.array([1]*M).reshape(1, M)
        x_tpu = self.controlled_process.next_step(1, x=x_t, u=u)
        self.assertAlmostEqual(np.mean(x_tpu), 1+self.controlled_process.dt, 2)
        self.assertEqual(np.shape(x_tpu)[1], M)

