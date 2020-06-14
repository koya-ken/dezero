import core.functions.function as f
import numpy as np


class Square(f.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray):
        x = self._input.data
        gx = 2 * x * gy
        return gx
