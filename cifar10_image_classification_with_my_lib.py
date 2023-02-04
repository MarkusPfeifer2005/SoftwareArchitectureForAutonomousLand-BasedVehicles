#!/usr/bin/env python3.10
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

from cifar10_knn_classification import Cifar10Dataset, evaluate
from my_machine_learning_library import SVMLossVectorized, LinearLayer,\
    SigmoidLayer, Model, StochasticGradientDecent
from configuration_handler import Config


class LinearClassifier(Model):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__(name="linear_classifier")
        self.file_prefix = "linear_classifier_params@"
        self.layers = [LinearLayer(num_pixels=num_pixels, num_classes=num_classes)]


class DoubleLinearClassifier(Model):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__(name="double_linear")
        self.file_prefix = "double_linear_classifier_params@"
        self.layers = [LinearLayer(num_pixels=num_pixels, num_classes=100),
                       LinearLayer(num_pixels=100, num_classes=num_classes)]


class SigmoidModel(Model):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__(name="sigmoid_classifier")
        self.file_prefix = "sigmoid_classifier@"
        self.layers = [SigmoidLayer(num_pixels=num_pixels, num_classes=1000),
                       LinearLayer(num_pixels=1000, num_classes=num_classes)]


def train(model: Model, dataset: Cifar10Dataset, criterion, optimizer, epochs: int, completed_epochs: int = 0,
          normalize: bool = False) -> list[float]:
    sleep(0.1)  # Ensure that print statements do not interfere.
    average_losses = []
    for _ in tqdm(range(completed_epochs, completed_epochs + epochs), desc="Training the model"):
        batch_losses = []
        for batch in dataset:
            data = batch[b"data"].astype("float64") / 255 if normalize else batch[b"data"].astype("float64")
            targets = batch[b"labels"]
            scores = model.forward(data)
            loss = criterion.forward(scores, targets)  # loss is the average loss over the entire batch (x)
            optimizer.step(grad=criterion.backward())
            batch_losses.append(loss)
        average_losses.append(sum(batch_losses)/len(batch_losses))
    return average_losses


def main():
    config = Config("config.json")

    train_set: Cifar10Dataset = Cifar10Dataset(batches=slice(0, 5), root=config["cifar"])
    test_set: Cifar10Dataset = Cifar10Dataset(batches=slice(5, 6), root=config["cifar"])
    epochs = 100

    models = [LinearClassifier(), DoubleLinearClassifier(), SigmoidModel()]
    criterion = SVMLossVectorized()

    for model in models:
        experiment_count = 5
        average_losses, accuracies, datasets = [], [], []
        for experiment in range(experiment_count):
            model.__init__()  # Ensure that new parameters are used for each experiment.
            optimizer = StochasticGradientDecent(model_layers=model.layers, lr=1e-3)
            average_losses.append(train(model, train_set, criterion, optimizer, epochs=epochs))
            accuracies.append(round(evaluate(model, train_set, normalize=True)))
            datasets.append(f"train #{experiment}")
            accuracies.append(round(evaluate(model, test_set, normalize=True)))
            datasets.append(f"test #{experiment}")

        # plotting:
        figure = plt.figure()

        axis1 = figure.add_subplot(1, 2, 1)
        axis1.set_title(f"Loss development of {model.name}.")
        axis1.set_ylabel("Average Losses")
        axis1.set_xlabel("Epochs")
        for experiment, average_loss in enumerate(average_losses):
            axis1.plot(average_loss, label=f"experiment #{experiment}")
        axis1.legend(loc="best")

        axis2 = figure.add_subplot(1, 2, 2)
        axis2.set_title(f"Accuracy of {model.name}.")
        axis2.set_ylabel("Accuracy in %")
        axis2.set_xlabel("Datasets")
        axis2.set_ylim([0, 100])
        axis2.bar(datasets, accuracies)
        for i in range(len(datasets)):
            axis2.text(i, accuracies[i], accuracies[i], ha="center")

        figure.tight_layout()  # Make sure labels are not overlapping.
        plt.show()


if __name__ == "__main__":
    main()
