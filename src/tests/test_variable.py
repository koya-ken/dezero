from unittest import TestCase

import numpy as np

from core import Variable


class TestVariable(TestCase):

    def test_initialize(self):
        data = np.array(1.0)
        x = Variable(data)
        self.assertEqual(x.data, data)
