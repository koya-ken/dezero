from typing import Optional

import numpy as np

import core.functions as f


class Variable:
    def __init__(self, data):
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[f.Function] = None

    def set_creator(self, func):
        self.creator = func
