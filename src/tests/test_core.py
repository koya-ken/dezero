from unittest import TestCase
from core.features import numerical_diff
from core import Variable
from core.functions import Square, Exp
import numpy as np


class TestCore(TestCase):

    def test_differential(self):
        f = Square()
        x = Variable(np.array(2.0))
        dy = numerical_diff(f, x)
        print(dy)

    def test_composite_function_differential(self):
        def f(x):
            A = Square()
            B = Exp()
            C = Square()
            return C(B(A(x)))

        x = Variable(np.array(0.5))
        dy = numerical_diff(f, x)
        print(dy)
