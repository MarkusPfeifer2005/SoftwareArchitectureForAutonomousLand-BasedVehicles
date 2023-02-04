#!/usr/bin/env python 3.10
from torchvision.datasets import MNIST
import numpy as np
from configuration_handler import Config


def get_losses(output_features: np.ndarray, labels: list[int], safety_margin: float = 1.) -> np.ndarray:
    losses = []
    for scores, target in zip(output_features, labels):
        penalty = np.maximum(0, scores - scores[target] + safety_margin)
        penalty[target] = 0
        losses.append(penalty.sum())
    losses = np.array(losses)
    return losses


class OversimplifiedLinearClassifier:
    def __init__(self):
        self.weight = np.random.random(size=(784, 10))
        self.bias = np.random.random(size=(10,))

    def forward(self, input_features: np.ndarray) -> np.ndarray:
        product = np.matmul(input_features, self.weight)
        output_features = product + self.bias
        return output_features


def main():
    config = Config("../config.json")
    np.set_printoptions(linewidth=4000, precision=3)

    dataset = MNIST(root=config["mnist"], download=False, train=True)
    model = OversimplifiedLinearClassifier()

    # create batch:
    batch_size = 64
    images, labels = [], []
    for index in range(batch_size):
        image, label = dataset[index]
        images.append(np.array(image).flatten())
        labels.append(label)
    batch = np.stack(images)

    probabilities = model.forward(input_features=batch)
    losses = get_losses(output_features=probabilities, labels=labels)
    print("Losses:")
    print(losses)
    print(f"Losses shape: {losses.shape}")

    model.weight[300][5] -= .1

    probabilities = model.forward(input_features=batch)
    losses = get_losses(output_features=probabilities, labels=labels)
    print("Losses:")
    print(losses)
    print(f"Losses shape: {losses.shape}")


if __name__ == "__main__":
    main()
