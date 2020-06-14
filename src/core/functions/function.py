import numpy as np

import core.variable as v


class Function:
    def __call__(self, value: 'v.Variable') -> 'v.Variable':
        x = value.data
        y = self.forward(x)
        output = v.Variable(y)
        output.set_creator(self)
        self._input: v.Variable = value
        self._output: v.Variable = output
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray):
        raise NotImplemented()
