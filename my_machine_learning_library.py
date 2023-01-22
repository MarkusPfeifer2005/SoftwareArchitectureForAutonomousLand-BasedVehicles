#!usr/bin/env python3.10
import os
import pickle
import numpy as np


class MathematicalFunc:
    """Abstract representation of a mathematical function.

    This is the core abstract class for this library. It is the code representation of a mathematical function.
    For this purpose it is outfitted with internal parameters and an input x. There can be multiple internal
    parameters, but only a single input.
    """

    def __init__(self, *args, **kwargs):
        self.parameters = []
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Representation of the execution of the mathematical function.

        The forward pass is the main pass of the mathematical function. Here the input is manipulated, using the
        internal parameters, according to the specified operations.

        :param x: ndarray
        :return: This method returns the output of the mathematical function, a ndarray.
        """

        raise NotImplementedError

    def backward(self, prev_grad: np.ndarray) -> tuple[np.ndarray, list]:
        """Representation of the partial derivative of the function.

        This is not the final derivative of the mathematical function, it is necessary to combine it using the chain
        rule. With the usage of the chain rule the gradient for the mathematical function parameters is calculated.

        :param prev_grad: The product of the previously calculated gradients. Chain rule was applied.
        :return: a tuple containing the ndarray holding the gradient for the input and a list holding the gradients of
        the parameters.
        """

        raise NotImplementedError


class Layer(MathematicalFunc):
    """Multiple nested mathematical functions.

    A sequence of nested mathematical functions forms a layer. From a mathematical point of view, a layer is simply a
    large mathematical function, but the abstraction of layers offers better usability. These layers are useful
    combinations of mathematical functions that perform specific tasks.
    """

    def __init__(self):
        super().__init__()
        self.operations = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """The input gets passed in sequence through all the forward methods of the operations."""

        for operation in self.operations:
            x = operation.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, list]:
        """Calculates the gradients for the input and the internal parameters.

        :param grad: Previous gradient.
        :return: Tuple holding the gradient for the function input and the gradients for all the parameters.
        """

        parameter_grads = []
        for operation in reversed(self.operations):
            grad, p_grads = operation.backward(grad)
            parameter_grads += reversed(p_grads)
        return grad, list(reversed(parameter_grads))


class Model:
    """Complete mathematical function modelling the training data.

    Multiple layers form a model. A model is the finalized mathematical function with and architecture adjusted to the
    training data.
    """

    def __init__(self, name: str = None):
        self.file_prefix = "model"
        self.name = name
        self.layers = []

    def forward(self, x: np.ndarray):
        """The input is passed through all layers of the model.

        :param x: Ndarray with the input of the model.
        :return: Output of the model.
        """

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def save(self, path: str, epoch: int):
        """Saves the whole model using pickle.

        :param path: A string holding the target address; the model name is automatically created.
        :param epoch: Number describing how often the model has been trained on the entire training dataset.
        """

        parameters = [layer.parameters for layer in self.layers]
        with open(os.path.join(path, f"{self.file_prefix}{epoch}"), "wb") as file:
            pickle.dump(parameters, file)

    def load(self, path: str) -> int:
        """ERROR - Dependencies of params of layers and params of m-functions are not resolved! Models do not learn!

        Loads weights and bias from file.

        The entire load/save interface is based on pickle.

        :param path: Path to the directory holding parameter files.
        :return epochs: int holding the number of epochs, that the model was already trained on.
        """

        epochs = [int(file.replace(self.file_prefix, '')) for file in os.listdir(path)
                  if file.startswith(self.file_prefix)]
        assert epochs != [], f"No files to load in {path}!"
        filename = os.path.join(path, f"{self.file_prefix}{str(max(epochs))}")

        with open(filename, "rb") as file:
            loaded_parameters = pickle.load(file)
        for layer, l_params in zip(self.layers, loaded_parameters):
            layer.parameters = l_params

        print(f"Loaded file '{filename}'.")
        return max(epochs)

    def __call__(self, x: np.ndarray):
        """Modified forward pass.

        :return: Index of the maximum in the output matrix.
        """

        return np.argmax(self.forward(x))

    def parameters(self) -> np.ndarray:
        """Gets all the parameters of the model.

        :return: Yields all the parameters.
        """

        for layer in self.layers:
            for op in layer.operations:
                for param in op.parameters:
                    yield param


class Optimizer:
    """Abstract class for optimizing models."""

    def __init__(self, model_layers: list, lr: float):
        self.lr = lr
        self.model_layers = model_layers


class Loss:
    """Computes the inaccuracy of the model.

    The loss function compares the target to the prediction of the model. It returns a numerical value representing the
    difference between prediction and target and therefore its inaccuracy.
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates difference between prediction and target.

        :param x: Predictions as ndarray.
        :param y: Targets as ndarray.
        :return: Numerical value for the inaccuracy of the model as a ndarray (depending on the function it can hold
        only a single element).
        """

        raise NotImplementedError

    def backward(self) -> np.ndarray:
        """Calculates gradient of the scores by applying the chain rule.

        :return: Gradient of the scores as ndarray.
        """

        raise NotImplementedError

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.forward(x, y)


class BiasAddition(MathematicalFunc):
    """Fundamental operation adding a bias matrix to the input."""

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
    """Fundamental operation multiplying a weight matrix to the input."""

    def __init__(self, weight: np.ndarray):
        super().__init__()
        self.parameters = [weight]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(self.x, self.parameters[0])

    def backward(self, prev_grad: np.ndarray) -> tuple[np.ndarray, list]:
        return np.dot(prev_grad, np.transpose(self.parameters[0])), [np.dot(np.transpose(self.x), prev_grad)]


class SVMLossVectorized(Loss):
    """Multiclass support vector machine loss."""

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
        return np.array(losses).sum() / self.x.size

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
    """Mean squared error loss."""

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
    """Weight multiplication and bias addition."""

    def __init__(self, num_pixels: int = 3072, num_classes: int = 10, weight_init: float = 1.):
        super().__init__()
        self.parameters: list[np.ndarray] = [np.random.uniform(-weight_init, weight_init, (num_pixels, num_classes)),
                                             np.random.uniform(-weight_init, weight_init, (num_classes,))]
        self.operations: list[MathematicalFunc] = [WeightMultiplication(weight=self.parameters[0]),
                                                   BiasAddition(bias=self.parameters[1])]


class SigmoidLayer(LinearLayer):
    """Linear layer with sigmoid activation function."""

    def __init__(self, num_pixels: int = 3072, num_classes: int = 10, weight_init: float = 1.):
        super().__init__(num_pixels=num_pixels, num_classes=num_classes, weight_init=weight_init)
        self.operations.append(Sigmoid())


class StochasticGradientDecent(Optimizer):
    def __init__(self, model_layers: list, lr: float, momentum: float = None):
        super().__init__(model_layers, lr)
        self.momentum = momentum
        if self.momentum:
            self.hist = [[np.zeros_like(param) for param in layer.parameters] for layer in reversed(self.model_layers)]

    def step(self, grad: np.ndarray):
        for l_idx, layer in enumerate(reversed(self.model_layers)):
            grad, parameter_grads = layer.backward(grad)
            for p_idx, (p_grad, param) in enumerate(zip(parameter_grads, layer.parameters)):
                if self.momentum:
                    delta = p_grad + self.momentum * self.hist[l_idx][p_idx]
                    self.hist[l_idx][p_idx] = delta
                    param -= delta * self.lr
                else:
                    param -= p_grad * self.lr
