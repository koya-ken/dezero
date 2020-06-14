import numpy as np
from typing import Optional


class Variable:
    def __init__(self, data):
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
