#!/usr/bin/env python3.10

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms

from init import Config
from benchmark.classification.torch_classification import train, ParentModel


class SemanticSegmentor(ParentModel):
    def __init__(self, image_size: tuple, in_channels: int, out_channels: int):
        super(SemanticSegmentor, self).__init__()
        # width, height = image_size
        # kernels = [5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
        # paddings = [kernel//2 for kernel in kernels]
        self.operations = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5, padding=2, stride=2),
            # torch.nn.Conv2d(in_channels=10, out_channels=50, kernel_size=3, padding=1, stride=2),
            # torch.nn.ConvTranspose2d(in_channels=50, out_channels=10, kernel_size=3, padding=1, stride=2),
            # torch.nn.ConvTranspose2d(in_channels=10, out_channels=out_channels, kernel_size=5, padding=2, stride=2),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.operations(x)
        return x


def evaluate(model,
             dataloader: torch.utils.data.DataLoader,
             device: str) -> float:
    """:return float: accuracy in %"""
    assert dataloader.batch_size == 1
    total = correct = 0
    model.eval()
    for image, target in dataloader:
        image, target = image.to(device), target.to(device)
        scores = model(image).argmax(dim=1)
        correct += (scores == target).to(torch.int8).sum().item()
        total += torch.numel(target)

    print(f"{correct} of {total} pixels were correct resulting in an accuracy of {correct/total*100:.2f}%.")
    return correct/total*100


class Mask:
    # All are sorted. No sets, because sets are not ordered.
    classes = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
        "void"
    ]
    colors = [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
        [220, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
        [255, 255, 255],
    ]
    colors = [tuple(color) for color in colors]
    class_indices = [i for i in range(len(classes))]
    class_indices[-1] = 255
    assert len(classes) == len(colors) == len(class_indices),\
        f"{len(classes) = }, {len(colors) = }, {len(class_indices) = }"

    def __init__(self, image: torch.Tensor):
        assert isinstance(image, torch.Tensor)
        image = transforms.ToPILImage()(image)
        self.image = Image.new("RGB", image.size)
        self.image.paste(image)

        for width in range(self.image.size[0]):
            for height in range(self.image.size[1]):
                masked_value = image.getpixel(xy=(width, height))
                if masked_value > len(self.classes) - 2:
                    masked_value = 255
                self.image.putpixel(xy=(width, height), value=self.colors[self.class_indices.index(masked_value)])

    def show(self):
        self.image.show()


def main():
    config = Config("config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = (375, 500)

    train_data = VOCSegmentation(root=config["voc-segmentation"],
                                 year="2012",
                                 image_set="train",
                                 download=False,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Resize(image_size)]),
                                 target_transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Resize(image_size),
                                                                      torch.squeeze]))
    train_loader = DataLoader(dataset=train_data,
                              batch_size=16,
                              shuffle=True)
    validation_data = VOCSegmentation(root=config["voc-segmentation"],
                                      year="2012",
                                      image_set="val",
                                      download=False,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Resize(image_size)]),
                                      target_transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Resize(image_size),
                                                                           torch.squeeze]))
    validation_dataloader = DataLoader(dataset=validation_data,
                                       batch_size=1,
                                       shuffle=True)

    model = SemanticSegmentor(image_size=image_size, in_channels=3, out_channels=22).to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

    evaluate(model=model,
             dataloader=validation_dataloader,
             device=device)

    train(model=model,
          dataloader=train_loader,
          criterion=torch.nn.CrossEntropyLoss(),
          optimizer=optimizer,
          epochs=3,
          show_graph=True,
          device=device)
    evaluate(model=model,
             dataloader=validation_dataloader,
             device=device)

    # Check a sample.
    model = model.to("cpu")
    model.eval()
    image, target = train_data[69]
    target = Mask(target)
    prediction = Mask(model(image).argmax(dim=0).to(torch.float32))
    prediction.show()
    target.show()


if __name__ == "__main__":
    main()
