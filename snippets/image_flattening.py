#!/usr/bin/env python 3.10
from torchvision.datasets import MNIST
import numpy as np
from configuration_handler import Config


def main():
    config = Config("../config.json")
    np.set_printoptions(linewidth=4000)

    # flatten image:
    dataset = MNIST(root=config["mnist"], download=False, train=True)
    image, label = dataset[0]
    image = np.array(image)
    print("Flattened image:")
    image = image.flatten()
    print(image)
    print(f"Flattened image shape: {image.shape}")


if __name__ == "__main__":
    main()
