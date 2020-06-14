import numpy as np

from core.functions.function import Function


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray):
        x = self._input.data
        gx = np.exp(x) * gy
        return gx
