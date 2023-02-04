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
    print(f"\nBatch shape: {batch.shape}")


if __name__ == "__main__":
    main()
