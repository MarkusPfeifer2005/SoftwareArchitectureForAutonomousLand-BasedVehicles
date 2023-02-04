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
    images = []
    for index in range(batch_size):
        image, label = dataset[index]
        image = np.array(image)
        image = image.flatten()
        images.append(image)
    batch = np.stack(images)
    weight = np.random.random(size=(784, 10)) * .001
    product = np.matmul(batch, weight)
    bias = np.random.random(size=(10,)) * .001
    output_features = product + bias

    # visualize output:
    print(output_features[:7])


if __name__ == "__main__":
    main()
