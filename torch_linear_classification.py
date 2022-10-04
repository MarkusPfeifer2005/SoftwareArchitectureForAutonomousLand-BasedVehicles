#!/usr/bin/env python
from nearest_neighbour import Cifar10Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn


def evaluate(model, dataset: Cifar10Dataset,  normalize: bool = False, images: slice = slice(None, None, None)) -> float:
    total = correct = 0
    for batch in dataset:
        for img, lbl in zip(batch[b"data"][images], batch[b"labels"][images]):
            img = img.astype("float64") / 255 if normalize else img.astype("float64")
            img = torch.from_numpy(img)
            target = dataset.labels[lbl].decode()
            scores = model(img).argmax().item()
            prediction = dataset.labels[scores].decode()

            total += 1
            if prediction == target:
                correct += 1

    print(f"{correct} of {total} examples were correct resulting in an accuracy of {correct/total*100:.2f}%.")
    return correct/total*100


class TorchLinearClassifier(nn.Module):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_pixels, out_features=num_classes)

    def forward(self, x):
        x = self.linear1(x)
        return x


class TorchExperimentalModel(nn.Module):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_pixels, out_features=100)
        self.linear2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TorchSigmoidModel(nn.Module):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_pixels, out_features=100)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x


def train(model: nn.Module, dataset: Cifar10Dataset, criterion, optimizer, epochs: int, completed_epochs: int = 0,
          normalize: bool = False):
    avg_losses = []
    for _ in tqdm(range(completed_epochs, completed_epochs + epochs), desc="Training the model"):
        batch_losses = []
        for batch in dataset:
            data = batch[b"data"].astype("float64") / 255 if normalize else batch[b"data"].astype("float64")
            targets = batch[b"labels"]
            data, targets = torch.from_numpy(data), torch.tensor(targets)

            scores = model.forward(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_losses.append(sum(batch_losses)/len(batch_losses))

    plt.plot(avg_losses)
    plt.ylabel("Average Losses")
    plt.xlabel("Epochs")
    plt.show()


def main():
    train_set: Cifar10Dataset = Cifar10Dataset(batches=slice(0, 1))
    evaluation_set: Cifar10Dataset = Cifar10Dataset(batches=slice(4, 5))
    test_set: Cifar10Dataset = Cifar10Dataset(batches=slice(5, 6))
    model = TorchLinearClassifier().double()
    # model = TorchExperimentalModel()
    # model = TorchSigmoidModel()
    parameters = [param.detach().numpy() for param in list(model.parameters())]
    criterion = nn.MultiMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = int(1e2)
    comp_epochs = 0
    train(model, train_set, criterion, optimizer, epochs=epochs, completed_epochs=comp_epochs)
    evaluate(model, train_set)
    evaluate(model, evaluation_set)
    # evaluate(model, test_set)


if __name__ == "__main__":
    main()
