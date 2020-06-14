from core.functions.function import Function
import numpy as np


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2
