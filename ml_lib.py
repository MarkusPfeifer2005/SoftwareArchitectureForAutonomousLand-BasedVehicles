import os
import pickle
import numpy as np

import torch
import torch.nn as nn


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
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x: np.ndarray, y: list[int]) -> np.ndarray:
        """
        l(S, Y) = sum(max(0, sj-sy+1)
        """
        self.x = x
        self.y = y
        losses = []
        for score, target in zip(self.x, self.y):
            margin = np.maximum(0, score - score[target] + 1)
            margin[target] = 0
            losses.append(margin.sum())  # TODO: Is this sum accounted for in the backward pass? /len(margin)?
        return np.array(losses).sum()/len(x)

    def backward(self) -> np.ndarray:
        """
        Returns a matrix of ones in the shape of the scores. Since the loss only consists of linear functions
        and a max-function only ones are passed on. The max-function is derived by zeroing the target
        element, since it has no effect on the loss.
        """
        grads = []
        for score, target in zip(self.x, self.y):
            #############
            # explanation needs to take sum(losses)/len(losses) into account!!!!
            #############

            # l(s) = sum(max(0, sj-st+1)  st is the target; it is not summed up
            # l(s) = c(b(a(s)))
            # a(s) = sj-st+1
            # b(s) = max(0, a(s)) https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
            # c(s) = sum(b(s))  st is not summed up

            # l'(s) = c'(b(a(s))) * b'(a(s)) * a'(s)

            # The derivative of the max function is not defined for 0 resulting in the spaghetti-code below...
            # da = np.ones_like(score)  # da/ds = 1  # OUTDATED!!! Replaced by line below:
            da = np.ones_like(score).astype("float64") / len(self.x)  # / len(score) = * len(score)^-1
            db = ((score - score[target] + 1.) >= 0.) * 1.
            # db[self.target] = 0  # can be ignored, because it is changed later
            # dc = np.ones_like(self.scores)
            grad = db * da  # dc * db * da     dc is a matrix of ones, so it can be ignored
            db = ((score - score[target] + 1.) > 0.) * 1.  # needs to be repeated with > instead of >=
            db[target] = 0.
            grad[target] = -sum(db) / len(self.x)  # TODO:  / len(self.x) requires further investigation!
            # assert grad.shape == score.shape  # for debugging
            grads.append(grad)
        return np.array(grads)


class SVMLossVectorizedII(SVMLossVectorized):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MultiMarginLoss()
        self.loss = None

    def forward(self, x: np.ndarray, y: list[int]) -> np.ndarray:
        self.x = torch.tensor(x, requires_grad=True)
        self.y = torch.tensor(y)
        self.loss = self.criterion(self.x, self.y)
        return self.loss.detach().numpy()

    def backward(self) -> np.ndarray:
        self.loss.backward()
        return self.x.grad.numpy()


# class L2Regularization(Loss):
#     def __init__(self):
#         self.weights = None
#
#     def forward(self, weights: np.ndarray) -> float:
#         self.weights = weights
#         penalty = self.weights**2
#         penalty = penalty.sum()
#         return penalty
#
#     def backward(self):
#         return self.weights * 2  # * np.ones_like(self.weights)


class LinearLayer(Layer):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.parameters = [np.random.uniform(-1, 1, (num_pixels, num_classes)),
                           np.random.uniform(-1, 1, (num_classes,))]
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
