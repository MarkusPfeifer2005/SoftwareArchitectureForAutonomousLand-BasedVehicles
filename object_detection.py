#!usr/bin/env python
import os
from PIL import Image, ImageDraw
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
        with open(os.path.join(self.root, "boxy_labels_train.json")) as file:
            self.labels = json.load(file)

    def __len__(self) -> int:
        """Number of images not of labels, so parts of dataset can be used."""
        return len(self.images)

    def __getitem__(self, idx: int) -> (torch.Tensor, list):
        img = Image.open(os.path.join(self.root, self.images[idx]))
        if self.transforms:
            img = self.transforms(img)
        # If the following code throws a KeyError, then change to second json file.
        aabbs = []  # No list comprehension for better readability.
        x_factor, y_factor = 1232/2464, 1028/2056  # Taking care of smaller image resolution.
        for vehicle in self.labels["./" + self.images[idx].replace('\\', '/')]["vehicles"]:
            aabbs.append([vehicle["AABB"]["x1"]*x_factor,
                          vehicle["AABB"]["y1"]*y_factor,
                          vehicle["AABB"]["x2"]*x_factor,
                          vehicle["AABB"]["y2"]*y_factor])
        return img, aabbs

    def show(self, idx: int):
        """Shows the image with the drawn bounding boxes."""
        img, bboxes = self.__getitem__(idx)
        img = transforms.ToPILImage()(img)  # Returns Image.
        painter = ImageDraw.Draw(img)
        for bbox in bboxes:
            painter.rectangle(bbox, outline=(0, 255, 0))
        img.show()


class VehicleDetector(torch.nn.Module):
    def __int__(self):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


def main():
    data = BoschDataset()


if __name__ == "__main__":
    main()
