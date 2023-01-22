#!/usr/bin/env python3.10
import torch
import torchvision
import torch.utils.data

from configuration_handler import Config
from mnist_image_classification_with_pytorch import SaveableModule, train, evaluate


class AlexNet(SaveableModule):
    """https://en.wikipedia.org/wiki/AlexNet"""
    def __init__(self, name: str = "alex_model"):
        super(AlexNet, self).__init__(name=name)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.max_pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = torch.nn.Linear(in_features=6400, out_features=4096)
        self.dropout1 = torch.nn.Dropout(p=.5)
        self.fc2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = torch.nn.Dropout(p=.5)
        self.fc3 = torch.nn.Linear(in_features=4096, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.max_pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def main():
    config = Config("../config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = (224, 224)  # Original Cifar-10 images are (32, 32).

    train_set = torchvision.datasets.CIFAR10(
        root=config["t-cifar"],
        download=False,  # Setting download to False saves time.
        train=True,  # Is automatically implemented, some files do not get used.
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=image_size)
        ])
    )
    test_set = torchvision.datasets.CIFAR10(
        root=config["t-cifar"],
        download=False,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=image_size)
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    alex_model = AlexNet().to(device)
    try:
        alex_model.load("../model-parameters/alex_model.pt")
    except FileNotFoundError:
        print("Could not load parameters, continue to use default values.")
    optimizer = torch.optim.Adam(alex_model.parameters(), lr=0.0009)
    train(
        model=alex_model,
        dataloader=train_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        epochs=5,
        show_graph=True,
        device=device
    )
    try:
        alex_model.save(path="../model-parameters")
    except OSError:
        pass
    evaluate(model=alex_model, dataloader=test_loader, device=device)


if __name__ == "__main__":
    main()
