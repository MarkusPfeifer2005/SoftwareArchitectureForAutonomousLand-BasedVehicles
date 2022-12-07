#!usr/bin/env python
import numpy as np
import torch

"""
All beta implementations!
"""

class BaseMatrix:
    def __init__(self, data, requires_grad: bool = False, op=None):
        self.requires_grad = requires_grad
        self.op = op

        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data
        elif isinstance(data, list) or isinstance(data, np.float64) or isinstance(data, float):
            self.data: np.ndarray = np.array(data)
        elif isinstance(data, torch.Tensor):
            self.data: np.ndarray = data.numpy()
        else:
            raise TypeError(f"Type {type(data)} is not supported for Matrix!")

    def toarray(self) -> np.ndarray:
        return self.data

    def tolist(self) -> object:
        return self.data.tolist()

    def backward(self):
        ...

class Function:
    ...

class Addition:
    def __init__(self):
        self.a = None
        self.b = None

    def forward(self, a, b):
        assert isinstance(a, BaseMatrix) and isinstance(b, BaseMatrix)
        self.a = a
        self.b = b
        result = self.a.data + self.b.data
        return BaseMatrix(data=result, requires_grad=(self.a.requires_grad or self.b.requires_grad), op=self)

    def backward(self, grad):
        raise NotImplementedError

class Summation:
    def __init__(self):
        self.a = None

    def forward(self, a):
        assert isinstance(a, BaseMatrix)
        self.a = a
        result = a.data.sum()
        return BaseMatrix(data=result, requires_grad=self.a.requires_grad, op=self)

    def backward(self):
        raise NotImplementedError

class Matrix(BaseMatrix):
    def __init__(self, data, requires_grad: bool = False, op=None):
        super().__init__(data=data, requires_grad=requires_grad, op=op)

    def __add__(self, other):
        return Matrix(self.data + other.data)

    def __sub__(self, other):
        return Matrix(self.data - other.data)

    def __mul__(self, other):
        return Matrix(self.data * other.data)

    def __truediv__(self, other):
        return Matrix(self.data / other.data)

    def __pow__(self, power, modulo=None):
        return Matrix(self.data.__pow__(power, modulo))

    def sum(self):
        return Matrix(self.data.sum())



if __name__ == "__main__":
    x = torch.tensor([[6., 3.], [5., 2.]], requires_grad=False)
    w = torch.tensor([[9., 5.], [7., 1.]], requires_grad=True)
    b = torch.tensor([[6., 5.], [9., 4.]], requires_grad=True)

    r = (x * w + b).sum()
    r.backward()

    x = Matrix([[6., 3.], [5., 2.]], requires_grad=False)
    w = Matrix([[9., 5.], [7., 1.]], requires_grad=True)
    b = Matrix([[6., 5.], [9., 4.]], requires_grad=True)

    r = (x * w + b).sum()
