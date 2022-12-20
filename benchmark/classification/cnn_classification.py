#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision
import torch.utils.data

from init import Config
from torch_classification import ParentModel, train, evaluate


class CNN(ParentModel):
    def __init__(self, num_classes: int = 10, name: str = None):
        super().__init__(name=name)
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
        )
        self.lin = nn.Sequential(
            nn.Linear(in_features=21632, out_features=num_classes),
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.reshape(x.shape[0], -1)
        x = self.lin(x)
        return x


def main():
    config = Config("../../config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = torchvision.datasets.CIFAR10(
        root=config["t-cifar"],
        download=False,  # Setting this to False saves time.
        train=True,  # Is automatically implemented, some files do not get used.
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
    )
    test_set = torchvision.datasets.CIFAR10(
        root=config["t-cifar"],
        download=False,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    model = CNN(name="CNN").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=0.75)
    train(
        model=model,
        dataloader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        epochs=5,
        show_graph=True,
        device=device,
    )
    evaluate(model, test_loader, device=device)
    model.save(path=r"..\..\model-parameters")


if __name__ == "__main__":
    main()
