#!/usr/bin/env python 3.10
from torchvision.datasets import MNIST
import numpy as np
from configuration_handler import Config


def main():
    config = Config("../config.json")
    np.set_printoptions(linewidth=4000)

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

    # create weight:
    weight = np.random.random(size=(784, 10))
    print(f"Weight shape: {weight.shape}")

    # multiply:
    product = np.matmul(batch, weight)
    print(f"Product shape: {product.shape}")


if __name__ == "__main__":
    main()
