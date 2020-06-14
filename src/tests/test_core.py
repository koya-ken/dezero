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

    def test_backward(self):
        A = Square()
        B = Exp()
        C = Square()
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        y.grad = np.array(1.0)
        b.grad = C.backward(y.grad)
        a.grad = B.backward(b.grad)
        x.grad = A.backward(a.grad)
        print(x.grad)
