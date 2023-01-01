#!/usr/bin/env python3.10

from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms

from init import Config


class PascalSemanticSegmentation:
    def __init__(self, root: str):
        self.root = root


def main():
    config = Config("config.json")

    train_data = VOCSegmentation(root=config["voc-segmentation"],
                                 year="2012",
                                 image_set="train",
                                 download=False,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Resize((375, 500))]),
                                 target_transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Resize((375, 500))]))
    train_loader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True)
    validation_data = VOCSegmentation(root=config["voc-segmentation"],
                                      year="2012",
                                      image_set="val",
                                      download=False,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Resize((375, 500))]),
                                      target_transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Resize((375, 500))]))
    validation_dataloader = DataLoader(dataset=validation_data,
                                       batch_size=32,
                                       shuffle=True)


if __name__ == "__main__":
    main()
