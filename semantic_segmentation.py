#!/usr/bin/env python3.10
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.utils import draw_segmentation_masks

from benchmark.classification.torch_classification import ParentModel, train
from init import Config


class PoolSemanticSegmentator(ParentModel):
    def __init__(self, image_channels: int, classes: int, name: str = "PoolSemanticSegmentator"):
        super(PoolSemanticSegmentator, self).__init__(name=name)
        # width, height = image_size
        # kernels = [5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
        # paddings = [kernel//2 for kernel in kernels]
        # (batch-size,) channels, height, width
        # 3, 375, 500
        self.conv1 = torch.nn.Conv2d(in_channels=image_channels, out_channels=10, kernel_size=5, padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxUnpool2d(kernel_size=2)
        self.conv4 = torch.nn.Conv2d(in_channels=10, out_channels=classes, kernel_size=5, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x, indices = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x, indices=indices)
        x = self.conv4(x)
        x = x[:, :, :375, :500]
        return x


class TransposeConvolutionSemanticSegmentator(ParentModel):
    def __init__(self, image_channels: int, classes: int, name: str = "TransposeConvolutionSemanticSegmentator"):
        super(TransposeConvolutionSemanticSegmentator, self).__init__(name=name)
        self.operations = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=image_channels, out_channels=5, kernel_size=7, padding=3),
            torch.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=2, padding=2),
            torch.nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=10, out_channels=32, kernel_size=5, stride=2, padding=2),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=10, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=5, stride=1, padding=2),
            torch.nn.ConvTranspose2d(in_channels=5, out_channels=5, kernel_size=5, stride=2, padding=2),
            torch.nn.ConvTranspose2d(in_channels=5, out_channels=classes, kernel_size=7, padding=0)  # Without padding.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.operations(x)
        x = x[:, :, :375, :500]
        return x


def evaluate(model,
             dataloader: torch.utils.data.DataLoader,
             device: str) -> float:
    """:return float: accuracy in %"""
    assert dataloader.batch_size == 1
    total = correct = 0
    model.eval()
    for image, target in tqdm(dataloader, desc="evaluating"):
        image, target = image.to(device), target.to(device)
        scores = model(image).argmax(dim=1)
        correct += (scores == target).to(torch.int8).sum().item()
        total += torch.numel(target)

    print(f"{correct} of {total} pixels were correct resulting in an accuracy of {correct / total * 100:.2f}%.")
    return correct / total * 100


class MaskedImage:
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
    assert len(classes) == len(colors) == len(class_indices), \
        f"{len(classes) = }, {len(colors) = }, {len(class_indices) = }"

    def __init__(self, image: torch.Tensor, mask: torch.Tensor):
        if len(image.shape) == 4:
            image = torch.squeeze(image, dim=0)  # (channel, height, width)
        if len(mask.shape) == 2:
            mask = torch.unsqueeze(mask, dim=0)  # (channel, height, width)
        attribution_maps = []
        for class_index in range(len(self.classes)):
            attribution_maps.append(mask == class_index)
        masks = torch.cat(attribution_maps, dim=0)
        self.image = draw_segmentation_masks(image=image, masks=masks)
        self.image = to_pil_image(self.image)

    def show(self):
        self.image.show()


def white_to_class_index(mask: torch.Tensor):
    """Reduce all pixels with the value 255 to the value 21."""
    twentyone = torch.full_like(input=mask, fill_value=21)
    mask = torch.where(mask <= 21, mask, twentyone)  # Condition is correct because it must be true so value is kept.
    assert torch.max(mask).item() <= 21
    return mask


def main():
    config = Config("config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = (375, 500)

    train_data = VOCSegmentation(root=config["voc-segmentation"],
                                 year="2012",
                                 image_set="train",
                                 download=False,
                                 transform=transforms.Compose([transforms.ToTensor(),  # Reduces range to 0 - 1.
                                                               transforms.Resize(image_size)]),
                                 # https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
                                 target_transform=transforms.Compose([pil_to_tensor,  # Seems to preserve range 0 - 255.
                                                                      transforms.Resize(image_size),
                                                                      torch.squeeze,  # size = (height, width)
                                                                      white_to_class_index]))  # classes range 0 - 21

    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    validation_data = VOCSegmentation(root=config["voc-segmentation"],
                                      year="2012",
                                      image_set="val",
                                      download=False,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Resize(image_size)]),
                                      target_transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Resize(image_size),
                                                                           torch.squeeze,
                                                                           white_to_class_index]))
    validation_dataloader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)

    # model = PoolSemanticSegmentator(image_channels=3, classes=22).to(device)
    model = TransposeConvolutionSemanticSegmentator(image_channels=3, classes=22).to(device)
    try:
        model.load("model-parameters/TransposeConvolutionSemanticSegmentator.pt")
    except FileNotFoundError:
        print("Could not load parameters, continue to use default values.")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0009)

    # Calculate class weights:
    # According to https://arxiv.org/pdf/1411.4038.pdf not necessary.
    # number_of_classes = 22
    # class_occurrences = [0. for _ in range(number_of_classes)]
    # for _, target in tqdm(train_data, desc="calculating class weights"):
    #     for class_index in range(number_of_classes):
    #         class_occurrences[class_index] += (target == class_index).sum().item()
    # class_weights = [1 / class_ for class_ in class_occurrences]

    train(model=model,
          dataloader=train_loader,
          criterion=torch.nn.CrossEntropyLoss(),
          optimizer=optimizer,
          epochs=20,
          show_graph=True,
          device=device)
    evaluate(model=model,
             dataloader=validation_dataloader,
             device=device)
    try:
        model.save(path="model-parameters")
    except OSError:
        pass

    # Check a sample.
    data = VOCSegmentation(root=config["voc-segmentation"],
                           year="2012",
                           image_set="val",
                           download=False,
                           transform=transforms.Compose([pil_to_tensor,  # Seems to preserve range 0 - 255.
                                                         transforms.Resize(image_size)]),
                           target_transform=transforms.Compose([pil_to_tensor,
                                                                transforms.Resize(image_size),
                                                                torch.squeeze,
                                                                white_to_class_index]))
    model = model.to("cpu")
    model.eval()
    image, target = data[0]
    image = image.unsqueeze(dim=0)  # Add a batch dimension.
    scores = model(image.to(torch.float32))
    scores = scores.argmax(dim=1)

    ground_truth = MaskedImage(image=image, mask=target)
    ground_truth.show()
    prediction = MaskedImage(image=image, mask=scores)
    prediction.show()


if __name__ == "__main__":
    main()
