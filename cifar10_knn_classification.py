#!/usr/bin/env python
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from configuration_handler import Config


class Cifar10Dataset:
    """For detailed information or download visit: http://www.cs.toronto.edu/~kriz/cifar.html

    structure:
    dataset
        batch
            batch_label: str
            labels: list[int]
            data: numpy.ndarray shape=(10000, 3072)
            filenames: list[bytes]
    """
    def __init__(self, root: str, batches: slice = slice(None, None, None), name: str = None):
        """
        :param root: str to the cifar directory folder.
        :param batches: slice; default uses the entire dataset.
        :param name: str giving the dataset a descriptive name.
        """
        self._root = root
        self.batches = batches
        self.name = name

        with open(os.path.join(root, "batches.meta"), "rb") as file:
            self.labels = pickle.load(file, encoding="bytes")[b"label_names"]

    def __iter__(self) -> object:
        """Yields content of batch (see docstring of MyDataset)."""
        for file in [file for file in os.listdir(self._root) if "_batch" in file][self.batches]:  # leaves out test data
            with open(os.path.join(self._root, file), "rb") as batch:
                yield pickle.load(batch, encoding="bytes")

    @property
    def num_classes(self) -> int:
        return len(self.labels)


class ManhattanModel:
    def __init__(self):
        self.data: np.ndarray = np.empty(shape=(0, 3072))
        self.labels: list = []

    def __call__(self, x: np.ndarray):
        """Finds the img with the least distance to the test img and use its label."""
        scores = np.array([img.sum() for img in abs(self.data - x)])
        return self.labels[np.argmin(scores)]


def train(model: ManhattanModel, dataset: Cifar10Dataset):
    """Puts the dataset values into the model."""
    for batch in dataset:
        model.data = np.append(model.data, batch[b"data"], axis=0)
        model.labels += batch[b"labels"]


def evaluate(model, dataset: Cifar10Dataset,
             show: bool = False,
             normalize: bool = False,
             images: slice = slice(None, None, None)) -> float:
    """Evaluates the model on the dataset and returns the accuracy in %."""
    total = correct = 0
    for batch in dataset:
        for img, lbl in zip(batch[b"data"][images], batch[b"labels"][images]):
            img = img.astype("float64")
            if normalize:
                img /= 250
            target = dataset.labels[lbl].decode()
            prediction = dataset.labels[model(img)].decode()

            if show:
                img = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
                plt.imshow(img)
                plt.title(f"prediction: {prediction}, target: {target}")
                plt.show()

            total += 1
            if prediction == target:
                correct += 1

    print(f"{correct} of {total} examples were correct, resulting in an accuracy of {correct/total*100:.2f}%.")
    return correct/total*100


def main():
    config = Config("config.json")
    train_set: Cifar10Dataset = Cifar10Dataset(batches=slice(0, 5), root=config["cifar"])
    test_set: Cifar10Dataset = Cifar10Dataset(batches=slice(5, 6), root=config["cifar"])
    model: ManhattanModel = ManhattanModel()

    train(model, train_set)
    evaluate(model, test_set, show=True, normalize=True)


if __name__ == "__main__":
    main()
