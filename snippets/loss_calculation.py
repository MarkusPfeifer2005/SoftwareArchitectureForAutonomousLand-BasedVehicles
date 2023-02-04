#!/usr/bin/env python 3.10
from torchvision.datasets import MNIST
import numpy as np
from configuration_handler import Config


def main():
    config = Config("../config.json")
    np.set_printoptions(linewidth=4000, precision=3)

    # create batch:
    dataset = MNIST(root=config["mnist"], download=False, train=True)
    batch_size = 64
    images, labels = [], []
    for index in range(batch_size):
        image, label = dataset[index]
        images.append(np.array(image).flatten())
        labels.append(label)
    batch = np.stack(images)
    weight = np.random.random(size=(784, 10))
    product = np.matmul(batch, weight)
    bias = np.random.random(size=(10,))
    output_features = product + bias

    # calculate losses:
    losses = []
    safety_margin = 1
    for scores, target in zip(output_features, labels):
        penalty = np.maximum(0, scores - scores[target] + safety_margin)
        penalty[target] = 0
        losses.append(penalty.sum())
    losses = np.array(losses)
    print("Losses:")
    print(losses)
    print(f"Losses shape: {losses.shape}")


if __name__ == "__main__":
    main()
