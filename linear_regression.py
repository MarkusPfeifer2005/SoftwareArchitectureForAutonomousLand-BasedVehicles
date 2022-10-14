#!/usr/bin/env python
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ml_lib import Model,  LinearLayer, StochasticGradientDecent, MSE


class LinerRegression(Model):
    def __init__(self, num_features: int, num_outputs: int):
        super().__init__()
        self.file_prefix = "linear_reg@"
        self.layers = [LinearLayer(num_pixels=num_features, num_classes=num_outputs, weight_init=1)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Runs the forward method."""
        return self.forward(x)


class GenerativeDataset:
    def __init__(self, func: callable, interval: slice):
        self.func = func
        self.ivl = interval

    def __iter__(self) -> tuple[np.ndarray, np.ndarray]:
        yield self.x, self.y

    @property
    def x(self) -> np.ndarray:
        return np.array([[float(x)] for x in range(self.ivl.start, self.ivl.stop, self.ivl.step)])

    @property
    def y(self) -> np.ndarray:
        return np.array([[float(self.func(x))] for x in range(self.ivl.start, self.ivl.stop, self.ivl.step)])


def train(model: Model, dataset: GenerativeDataset, criterion, optimizer, epochs: int, completed_epochs: int = 0,
          sax: plt.subplot = None, lax: plt.subplot = None):
    avg_losses = []
    for _ in tqdm(range(completed_epochs, completed_epochs + epochs), desc="Training the model"):
        batch_losses = []
        for batch in dataset:
            data, targets = batch

            scores = model.forward(data)
            sax.plot(scores)
            loss = criterion.forward(scores, targets)  # loss is the average loss over the entire batch (x)
            optimizer.step(grad=criterion.backward())
            batch_losses.append(loss)
        avg_losses.append(sum(batch_losses)/len(batch_losses))

    if lax:
        lax.plot(avg_losses)


def main():
    dataset = GenerativeDataset(lambda x: -2 * x + 30, slice(0, 100, 1))
    model = LinerRegression(num_features=1, num_outputs=1)
    optim = StochasticGradientDecent(model.layers, lr=1e-4)
    loss_func = MSE()

    # create frame
    fig = plt.figure()

    # create plot 1
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("gradient descent")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # create plot 2
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("loss development")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")

    # 3d plot of loss
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.set_title("l(f(w, b))")
    ax3.set_xlabel('w')
    ax3.set_ylabel('b')
    ax3.set_zlabel('l')

    # generate data
    train(model=model, dataset=dataset, criterion=loss_func, optimizer=optim, epochs=int(50), lax=ax2, sax=ax1)
    ax1.scatter(dataset.x, dataset.y, s=1)
    for w in np.arange(-3, 0, .1):
        for b in np.arange(-3, 0, .1):
            model.layers[0].parameters = [np.array([[w]]).astype("float64"), np.array([[b]]).astype("float64")]
            model.layers[0].operations[0].parameters[0] = model.layers[0].parameters[0]  # restore reference
            model.layers[0].operations[1].parameters[0] = model.layers[0].parameters[1]  # restore reference
            ax3.scatter(w, b, loss_func(model(dataset.x), dataset.y), color="green", s=5)

    # Plotting the graphs.
    fig.tight_layout()  # Make sure labels are not overlapping.
    plt.show()


if __name__ == "__main__":
    main()
