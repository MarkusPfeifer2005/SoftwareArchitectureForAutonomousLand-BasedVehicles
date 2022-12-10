#!/usr/bin/env python
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torch.utils.data

from init import Config


class ParentModel(nn.Module):
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name


class TorchLinearClassifier(ParentModel):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10, name: str = None):
        super().__init__(name=name)
        self.linear1 = nn.Linear(in_features=num_pixels, out_features=num_classes)

    def forward(self, x):
        x = self.linear1(x)
        return x


class TorchExperimentalModel(ParentModel):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10, name: str = None):
        super().__init__(name=name)
        self.linear1 = nn.Linear(in_features=num_pixels, out_features=100)
        self.linear2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TorchSigmoidModel(ParentModel):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10, name: str = None):
        super().__init__(name=name)
        self.linear1 = nn.Linear(in_features=num_pixels, out_features=100)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x


class AlphaModel(ParentModel):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10, name: str = None):
        super().__init__(name=name)
        self.operations = nn.Sequential(
            nn.Linear(in_features=num_pixels, out_features=100),
            nn.Tanh(),
            nn.Linear(in_features=100, out_features=num_classes),
        )

    def forward(self, x):
        return self.operations(x)


def train_torch_mnist(model: ParentModel,
                      dataloader: torch.utils.data.DataLoader,
                      criterion,
                      optimizer,
                      epochs: int,
                      completed_epochs: int = 0,
                      show_graph: bool = True,
                      device: str = "cpu",
                      scheduler=None):
    """For MNIST and torch only."""
    model.train()
    avg_losses = []
    for _ in tqdm(range(completed_epochs, completed_epochs + epochs), desc=f"Training '{model.name}'"):
        batch_losses = []
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            data = data.squeeze(dim=1)
            data = data.flatten(start_dim=1)
            scores = model.forward(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_losses.append(sum(batch_losses)/len(batch_losses))
        if scheduler:
            scheduler.step()

    if show_graph:
        plt.plot(avg_losses)
        plt.title(f"Stats for Model '{model.name}'")
        plt.ylabel("Average Losses")
        plt.xlabel("Epochs")
        plt.show()


def evaluate_torch_mnist(model,
             dataset,
             device: str = "cpu") -> float:
    """For MNIST and torch only."""
    total = correct = 0
    model.eval()
    for data, targets in dataset:
        data, targets = data.to(device), targets.to(device)
        data = data.squeeze(dim=1)
        data = data.flatten(start_dim=1)
        for img, lbl in zip(data, targets):
            scores = model(img).argmax().item()
            total += 1
            if scores == lbl:
                correct += 1

    print(f"{correct} of {total} examples were correct resulting in an accuracy of {correct/total*100:.2f}%.")
    return correct/total*100


def main():
    """Testing the performance of the AlphaModel on the MNIST dataset."""
    config = Config("../config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = torchvision.datasets.MNIST(
        root=config["mnist"],
        download=True,
        train=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_set, test_set = torch.utils.data.random_split(dataset, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    model = AlphaModel(num_pixels=784, name="Alpha").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.75)
    train_torch_mnist(
        model=model,
        dataloader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        epochs=10,
        show_graph=True,
        device=device,
    )
    evaluate_torch_mnist(model, test_loader, device=device)


if __name__ == "__main__":
    main()
