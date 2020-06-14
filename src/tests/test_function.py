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

    def test_function_chain(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        self.assertTrue(y.creator == C)
        self.assertTrue(y.creator._input == b)
        self.assertTrue(y.creator._input.creator == B)
        self.assertTrue(y.creator._input.creator._input == a)
        self.assertTrue(y.creator._input.creator._input.creator == A)
        self.assertTrue(y.creator._input.creator._input.creator._input == x)

    def test_function_chain_backward(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        y.grad = np.array(1.0)
        C = y.creator
        b = C._input
        b.grad = C.backward(y.grad)
        B = b.creator
        a = B._input
        a.grad = B.backward(b.grad)

        A = a.creator
        x = A._input
        x.grad = A.backward(a.grad)
        print(x.grad)
