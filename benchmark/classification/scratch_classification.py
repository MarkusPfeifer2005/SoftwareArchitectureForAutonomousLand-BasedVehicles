#!/usr/bin/env python
import matplotlib.pyplot as plt
from tqdm import tqdm

from benchmark.nearest_neighbour import Cifar10Dataset, evaluate
from mlib.scratch import SVMLossVectorized, LinearLayer, SigmoidLayer, Model, StochasticGradientDecent
from init import Config


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
    config = Config("../../config.json")

    train_set: Cifar10Dataset = Cifar10Dataset(batches=slice(0, 4), root=config["cifar"])
    evaluation_set: Cifar10Dataset = Cifar10Dataset(batches=slice(4, 5), root=config["cifar"])
    test_set: Cifar10Dataset = Cifar10Dataset(batches=slice(5, 6), root=config["cifar"])
    model = SigmoidModel()
    criterion = SVMLossVectorized()
    optimizer = StochasticGradientDecent(model_layers=model.layers, lr=1e-3)

    epochs = 3
    # Loading does not work (see message in Model.load doc)!
    # if input("Do you want to try and load model parameters? (y/n): ").lower() == 'y':
    #     comp_epochs = model.load(config["model-parameters"])
    #     print(f"Loaded model parameters pretrained on {comp_epochs} epochs.")
    # else:
    #     comp_epochs = 0
    #     print("Did not load models.")
    train(model, train_set, criterion, optimizer, epochs=epochs)
    # model.save(path=config["model-parameters"], epoch=epochs + comp_epochs)
    evaluate(model, train_set, normalize=True)
    evaluate(model, evaluation_set, normalize=True)
    evaluate(model, test_set, normalize=True)


if __name__ == "__main__":
    main()
