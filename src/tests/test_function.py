from unittest import TestCase
from core import Variable
from core.functions import Function, Square, Exp
import numpy as np


class TestFunction(TestCase):

    def test_function(self):
        x = Variable(np.array(10))
        f = Square()
        y = f(x)
        self.assertTrue(isinstance(y, Variable))
        print(y.data)

    def test_composite(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        print(y.data)