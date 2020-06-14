from unittest import TestCase
from core.features import numerical_diff
from core import Variable
from core.functions import Square
import numpy as np


class TestCore(TestCase):

    def test_differential(self):
        f = Square()
        x = Variable(np.array(2.0))
        dy = numerical_diff(f, x)
        print(dy)