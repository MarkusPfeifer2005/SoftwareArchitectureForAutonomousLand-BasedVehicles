#!/usr/bin/env python
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ml_lib import Model,  LinearLayer, StochasticGradientDecent, MSE


class LinerRegression(Model):
    def __init__(self, num_features: int, num_outputs: int):
        super().__init__()
        self.file_prefix = "linear_reg@"
        self.layers = [LinearLayer(num_pixels=num_features, num_classes=num_outputs, weight_init=1.)]

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
          axs: dict[plt.subplot, ...] = None):
    """dict keys: gr_dc, ls_dv, gb_ls, wb_dv"""
    avg_losses, weights, biases = [], [], []
    for epoch in tqdm(range(completed_epochs, completed_epochs + epochs), desc="Training the model"):
        batch_losses = []
        for batch in dataset:
            data, targets = batch
            scores = model.forward(data)
            loss = criterion.forward(scores, targets)  # loss is the average loss over the entire batch (x)
            optimizer.step(grad=criterion.backward())
            if axs["gr_dc"]:  # plotting
                axs["gr_dc"].plot(scores, label=f"epoch: {epoch}")
            batch_losses.append(loss)
        weights.append(model.layers[0].parameters[0].item())
        biases.append(model.layers[0].parameters[1].item())
        avg_loss = sum(batch_losses)/len(batch_losses)
        avg_losses.append(avg_loss)
        if axs["gb_ls"]:  # plotting
            axs["gb_ls"].scatter(model.layers[0].parameters[0], model.layers[0].parameters[1],
                                 avg_loss, color="red", s=5)

    # plotting:
    if axs["ls_dv"]:
        axs["ls_dv"].plot(avg_losses)
    if axs["wb_dv"]:
        axs["wb_dv"].plot(weights)
        axs["wb_dv"].plot(biases)

    axs["gr_dc"].legend(loc="upper right", prop={'size': 3})


def main():
    dataset = GenerativeDataset(lambda x: -2 * x + 7, slice(0, 100, 1))
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

    # weight and bias development
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("parameter development")
    ax4.set_xlabel("epoch")
    ax4.set_ylabel("parameter")

    # generate data
    train(model=model, dataset=dataset, criterion=loss_func, optimizer=optim, epochs=int(30),
          axs={"gr_dc": ax1, "ls_dv": ax2, "gb_ls": ax3, "wb_dv": ax4})
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
