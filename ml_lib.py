import os
import pickle
import numpy as np


class MathematicalFunc:
    def __init__(self, *args, **kwargs):
        self.parameters = []
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, prev_grad: np.ndarray) -> tuple[np.ndarray, list]:
        raise NotImplementedError


class Layer(MathematicalFunc):
    def __init__(self):
        super().__init__()
        self.operations = []

    def forward(self, x) -> np.ndarray:
        for operation in self.operations:
            x = operation.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, list]:
        parameter_grads = []
        for operation in reversed(self.operations):
            grad, p_grads = operation.backward(grad)
            parameter_grads += reversed(p_grads)
        return grad, list(reversed(parameter_grads))


class Model:
    def __init__(self):
        self.file_prefix = "model"
        self.layers = []

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def save(self, path: str, epoch: int):
        """Saves the whole model using pickle."""
        parameters = [layer.parameters for layer in self.layers]
        with open(os.path.join(path, f"{self.file_prefix}{epoch}"), "wb") as file:
            pickle.dump(parameters, file)

    def load(self, path: str) -> int:
        """Loads weights and bias from file."""
        epochs = [int(file.replace(self.file_prefix, '')) for file in os.listdir(path) if self.file_prefix in file]
        assert epochs != [], f"No files to load in {path}!"
        filename = os.path.join(path, f"{self.file_prefix}{str(max(epochs))}")

        with open(filename, "rb") as file:
            parameters = pickle.load(file)
        for layer, feature in zip(self.layers, parameters):
            layer.parameters = parameters

        print(f"Loaded file '{filename}'.")
        return max(epochs)

    def __call__(self, x: np.ndarray):
        return np.argmax(self.forward(x))

    def parameters(self) -> np.ndarray:
        for layer in self.layers:
            for op in layer.operations:
                for param in op.parameters:
                    yield param


class Optimizer:
    def __init__(self, model_layers: list, lr: float):
        self.lr = lr
        self.model_layers = model_layers


class Loss:
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.forward(x, y)


class BiasAddition(MathematicalFunc):
    def __init__(self, bias: np.ndarray):
        super().__init__()
        self.parameters = [bias]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return self.x + self.parameters[0]

    def backward(self, prev_grad: np.ndarray) -> tuple[np.ndarray, list]:
        return prev_grad, [prev_grad.sum(axis=0)]  # Sum due to the one thing being only a vector (see unittest).


class Sigmoid(MathematicalFunc):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return 1 / (1 + np.exp(-self.x))

    def backward(self, prev_grad) -> tuple[np.ndarray, list]:
        return (self.forward(self.x) * (1 - self.forward(self.x))) * prev_grad, []


class WeightMultiplication(MathematicalFunc):
    def __init__(self, weight: np.ndarray):
        super().__init__()
        self.parameters = [weight]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(self.x, self.parameters[0])

    def backward(self, prev_grad: np.ndarray) -> tuple[np.ndarray, list]:
        return np.dot(prev_grad, np.transpose(self.parameters[0])), [np.dot(np.transpose(self.x), prev_grad)]


class SVMLossVectorized(Loss):
    def __init__(self, margin: float = 1):
        self.margin = margin
        self.x = None
        self.y = None

    def forward(self, x: np.ndarray, y: list[int]) -> float:
        """
        x are the scores of the model
        y are the indices of the targets
        l(X,Y) = sum(max(0, X-Y+m)) / len(X)
        """
        from time import perf_counter
        self.x = x

        # self.y = np.array([[row[idx]] for row, idx in zip(x, y)])
        # - margin to compensate for the loss of the true prediction wich is max(0, x - x + 1) and  adds 1 to the loss.
        # return np.sum(np.maximum(np.zeros_like(self.x), (self.x - self.y + self.margin)))/len(self.x)-self.margin

        # The following code is faster, than the one in the comment!
        self.y = y
        losses = []
        for score, target in zip(self.x, self.y):
            penalty = np.maximum(0, score - score[target] + self.margin)
            penalty[target] = 0
            losses.append(penalty.sum())
        return np.array(losses).sum()/self.x.size

    def backward(self) -> np.ndarray:
        """
        Returns a matrix of ones in the shape of the scores. Since the loss only consists of linear functions
        and a max-function only ones are passed on. The max-function is derived by zeroing the target
        element, since it has no effect on the loss.
        For the derivative of the max-function see:
        https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy

        l(x,y) = sum(max(0, x-y+m)) / len(x) - m
        l(x, y) = d(c(b(a(x, y))))
        a(x, y) = x-y+m
        b(a) = max(0, a)
        c(b) = sum(b)
        d(c) = c / len(c)

        dl/dd = 1
        dl/dc = -1 / len(c) * ld/dd
        dl/db = ones_like(b) * dl/dc
        dl/da = 1 if a > 0 else 0 * dl/db
        dl/dx = ones_like(x) * dl/da
        """
        # y = np.array([[row[idx]] for row, idx in zip(self.x, self.y)])
        # dldc: float = -1. / pow(np.sum(np.maximum(np.zeros_like(self.x), (self.x - self.y + self.margin))), 2)
        # dldb = np.ones_like(np.maximum(np.zeros_like(self.x), (self.x - self.y + self.margin))) * dldc
        # dlda = ((self.x - self.y + self.margin) > 0).astype("float64") * dldb
        # dldx = np.ones_like(self.x) * dlda
        # Does not account for the increased leverage of the target!

        dldc: float = 1. / self.x.size
        dldb = np.ones_like(self.x) * dldc  # replace with full

        max_grad = np.zeros_like(self.x)
        for x in range(max_grad.shape[0]):
            for y in range(max_grad.shape[1]):
                max_grad[x][y] = (self.x[x][y] - self.x[x][self.y[x]] + self.margin) > 0

        dldb = np.multiply(dldb, max_grad)

        for row, idx in zip(dldb, self.y):
            row[idx] = -(row.sum() - row[idx])

        return np.array(dldb)


class MSE(Loss):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """l(X, Y) = sum((X - Y)^2) / len(X)"""
        self.x, self.y = x, y
        return np.mean(np.power(self.x - self.y, 2))

    def backward(self) -> np.ndarray:
        """
        l(X, Y) = d(c(b(a(X, Y))))
        a(X, Y) = X - Y
        b(a) = a^2
        c(b) = sum(b)
        d(c) = c / len(X)

        d'(c) = 1 / len(X)
        c'(b) = ones_like(X)
        b'(a) = 2 * a
        a'(X) = ones_like(X)
        """
        return 1 / len(self.x) * np.ones_like(self.x) * 2 * (self.x - self.y) * np.ones_like(self.x)


class LinearLayer(Layer):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10, weight_init: float = 1.):
        super().__init__()
        self.parameters = [np.random.uniform(-weight_init, weight_init, (num_pixels, num_classes)),
                           np.random.uniform(-weight_init, weight_init, (num_classes,))]
        self.operations = [WeightMultiplication(weight=self.parameters[0]), BiasAddition(bias=self.parameters[1])]


class SigmoidLayer(Layer):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.parameters = [np.random.uniform(-1, 1, (num_pixels, num_classes)),
                           np.random.uniform(-1, 1, (num_classes,))]
        self.operations = [WeightMultiplication(weight=self.parameters[0]),
                           BiasAddition(bias=self.parameters[1]),
                           Sigmoid()]


class StochasticGradientDecent(Optimizer):
    def __init__(self, model_layers: list, lr: float):
        super().__init__(model_layers, lr)

    def step(self, grad: np.ndarray):
        for layer in reversed(self.model_layers):
            grad, parameter_grads = layer.backward(grad)
            for p_grad, param in zip(parameter_grads, layer.parameters):
                param -= p_grad * self.lr
