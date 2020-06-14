from unittest import TestCase
from core import Variable
from core.functions import Function, Square
import numpy as np


class TestFunction(TestCase):

    def test_function(self):
        x = Variable(np.array(10))
        f = Square()
        y = f(x)
        self.assertTrue(isinstance(y, Variable))
        print(y.data)
