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
    dataset = GenerativeDataset(lambda x: -2 * x + 3, slice(0, 100, 1))
    model = LinerRegression(num_features=1, num_outputs=1)
    optim = StochasticGradientDecent(model.layers, lr=1e-4)

    fig, axs = plt.subplots(2)
    fig.tight_layout()
    axs[0].set_title("gradient descent")
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[1].set_title("loss development")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("loss")

    train(model=model, dataset=dataset, criterion=MSE(), optimizer=optim, epochs=int(1e2), lax=axs[1], sax=axs[0])

    axs[0].scatter(dataset.x, dataset.y, s=1)
    plt.show()


if __name__ == "__main__":
    main()
