#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision
import torch.utils.data

from init import Config
from torch_classification import ParentModel, train, evaluate


class Classic(ParentModel):
    def __init__(self, num_pixels: int = 784, num_classes: int = 10, name: str = None):
        super().__init__(name=name)
        self.operations = nn.Sequential(
            nn.Linear(in_features=num_pixels, out_features=100),
            nn.Tanh(),
            nn.Linear(in_features=100, out_features=num_classes),
        )

    def forward(self, x):
        return self.operations(x)


class CNN(ParentModel):
    def __init__(self, num_classes: int = 10, name: str = None):
        super().__init__(name=name)
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5),
            nn.Tanh(),
        )
        self.lin = nn.Sequential(
            nn.Linear(in_features=6912, out_features=num_classes),
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.reshape(x.shape[0], -1)
        x = self.lin(x)
        return x


def main():
    config = Config("../../config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set1 = torchvision.datasets.MNIST(
        root=config["mnist"],
        download=False,  # Setting this to False saves time.
        train=True,  # Is automatically implemented, some files do not get used.
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torch.flatten,
        ]),
    )
    test_set1 = torchvision.datasets.MNIST(
        root=config["mnist"],
        download=False,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torch.flatten,
        ]),
    )
    train_set2 = torchvision.datasets.MNIST(
        root=config["mnist"],
        download=False,  # Setting this to False saves time.
        train=True,  # Is automatically implemented, some files do not get used.
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
    )
    test_set2 = torchvision.datasets.MNIST(
        root=config["mnist"],
        download=False,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
    )
    train_loader1 = torch.utils.data.DataLoader(train_set1, batch_size=32, shuffle=True)
    test_loader1 = torch.utils.data.DataLoader(test_set1, batch_size=1, shuffle=True)
    train_loader2 = torch.utils.data.DataLoader(train_set2, batch_size=32, shuffle=True)
    test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size=1, shuffle=True)
    classic_model = Classic(name="Classic").to(device)
    cnn_model = CNN(name="CNN").to(device)
    optimizer1 = torch.optim.SGD(classic_model.parameters(), lr=.005, momentum=0.75)
    optimizer2 = torch.optim.SGD(cnn_model.parameters(), lr=.005, momentum=0.75)
    train(
        model=classic_model,
        dataloader=train_loader1,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer1,
        epochs=5,
        show_graph=False,
        device=device,
    )
    train(
        model=cnn_model,
        dataloader=train_loader2,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer2,
        epochs=5,
        show_graph=False,
        device=device,
    )
    print(f"Evaluating model {classic_model.name}:")
    evaluate(classic_model, test_loader1, device=device)
    print(f"Evaluating model {cnn_model.name}:")
    evaluate(cnn_model, test_loader2, device=device)


if __name__ == "__main__":
    main()
