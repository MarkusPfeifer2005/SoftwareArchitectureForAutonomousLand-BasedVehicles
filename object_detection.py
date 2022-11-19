#!usr/bin/env python
import os
from PIL import Image
import json
from main import Config
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BoschDataset(Dataset):
    def __init__(self, transformation=transforms.Compose([transforms.PILToTensor()]), root: str = Config()["boxy"]):
        self.root = root
        self.transforms = transformation
        self.images = []  # No list comprehension for better maintainability.
        for directory in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, directory)):
                for file in os.listdir(os.path.join(self.root, directory)):
                    if ".png" in file:
                        self.images.append(os.path.join(directory, file))
        with open(os.path.join(self.root, "boxy_labels_valid.json")) as file:
            self.labels = json.load(file)

    def __len__(self) -> int:
        """Number of images not of labels, so parts of dataset can be used."""
        return len(self.images)

    def __getitem__(self, idx: int) -> (torch.Tensor, list[dict]):
        img = Image.open(os.path.join(self.root, self.images[idx]))
        if self.transforms:
            img = self.transforms(img)
        return img, self.labels["./" + self.images[idx].replace('\\', '/')]["vehicles"]


def main():
    data = BoschDataset()
    for item in data:
        print(item)


if __name__ == "__main__":
    main()
