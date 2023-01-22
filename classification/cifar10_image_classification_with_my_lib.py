#!/usr/bin/env python3.10
import matplotlib.pyplot as plt
from tqdm import tqdm

from cifar10_knn_classification import Cifar10Dataset, evaluate
from my_machine_learning_library import SVMLossVectorized, LinearLayer,\
    SigmoidLayer, Model, StochasticGradientDecent
from configuration_handler import Config


class LinearClassifier(Model):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.file_prefix = "linear_classifier_params@"
        self.layers = [LinearLayer(num_pixels=num_pixels, num_classes=num_classes)]


class DoubleLinearClassifier(Model):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.file_prefix = "double_linear_classifier_params@"
        self.layers = [LinearLayer(num_pixels=num_pixels, num_classes=100),
                       LinearLayer(num_pixels=100, num_classes=num_classes)]


class SigmoidModel(Model):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.file_prefix = "sigmoid_model@"
        self.layers = [SigmoidLayer(num_pixels=num_pixels, num_classes=1000),
                       LinearLayer(num_pixels=1000, num_classes=num_classes)]


def train(model: Model, dataset: Cifar10Dataset, criterion, optimizer, epochs: int, completed_epochs: int = 0,
          normalize: bool = False):
    avg_losses = []
    for _ in tqdm(range(completed_epochs, completed_epochs + epochs), desc="Training the model"):
        batch_losses = []
        for batch in dataset:
            data = batch[b"data"].astype("float64") / 255 if normalize else batch[b"data"].astype("float64")
            targets = batch[b"labels"]
            scores = model.forward(data)
            loss = criterion.forward(scores, targets)  # loss is the average loss over the entire batch (x)
            optimizer.step(grad=criterion.backward())
            batch_losses.append(loss)
        avg_losses.append(sum(batch_losses)/len(batch_losses))

    plt.plot(avg_losses)
    plt.ylabel("Average Losses")
    plt.xlabel("Epochs")
    plt.show()


def main():
    config = Config("../config.json")

    train_set: Cifar10Dataset = Cifar10Dataset(batches=slice(0, 5), root=config["cifar"])
    test_set: Cifar10Dataset = Cifar10Dataset(batches=slice(5, 6), root=config["cifar"])
    epochs = 10

    models = [LinearClassifier(), DoubleLinearClassifier(), SigmoidModel()]
    criterion = SVMLossVectorized()

    for model in models:
        optimizer = StochasticGradientDecent(model_layers=model.layers, lr=1e-3)
        train(model, train_set, criterion, optimizer, epochs=epochs)
        evaluate(model, train_set, normalize=True)
        evaluate(model, test_set, normalize=True)


if __name__ == "__main__":
    main()
