from core.variable import Variable
import numpy as np


class Function:
    def __call__(self, value: Variable) -> Variable:
        x = value.data
        y = self.forward(x)
        output = Variable(y)
        self._input: Variable = value
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray):
        raise NotImplemented()
