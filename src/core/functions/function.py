from core.variable import Variable
import numpy as np


class Function:
    def __call__(self, value: Variable) -> Variable:
        x = value.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
