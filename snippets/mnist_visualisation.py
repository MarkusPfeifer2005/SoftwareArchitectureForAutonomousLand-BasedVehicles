#!/usr/bin/env python3.10
from torchvision.datasets import MNIST
import numpy as np
from configuration_handler import Config


def main():
    config = Config("../config.json")  # get the mnist directory
    np.set_printoptions(linewidth=1000)

    dataset = MNIST(root=config["mnist"], download=False, train=True)

    image, label = dataset[0]  # load the first sample of the dataset
    print(np.array(image))
    print(f"This image shows a/an {label}.")
    image.show()
    # image.save(f"../MNIST_{label}.png")


if __name__ == "__main__":
    main()
