#!usr/bin/env python
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
from torchvision.ops import generalized_box_iou_loss
from torchvision.utils import draw_bounding_boxes

from init import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
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
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "void"
]


def save(model: torch.nn.Module, path: str, name: str = "FasterRCNN"):
    """Saves model to file. Extension: '.pt'."""
    path = os.path.join(path, f"{name}.pt")
    torch.save(model.state_dict(), path)
    print(f"Model weights saved successfully to {path}.")
    time.sleep(0.1)  # Keep print statements apart.


def load(model: torch.nn.Module, path: str):
    model.load_state_dict(torch.load(path))
    print("Model weights loaded successfully.")
    time.sleep(0.1)  # Keep print statements apart.


class AnnotatedImage:
    def __init__(self, image: torch.Tensor, annotations: list):
        assert len(annotations) == 1  # Only one input image.
        annotation = annotations[0]

        image = image.clone()
        image *= 255
        self.image = image.to(torch.uint8)
        self.boxes = annotation["boxes"]
        self.labels = [classes[label.item()] for label in annotation["labels"]]

    def show(self):
        image = draw_bounding_boxes(image=self.image, boxes=self.boxes, labels=self.labels)
        image = to_pil_image(image)
        image.show()


def train(model: torch.nn.Module,
          data_loader: DataLoader,
          optimizer,
          epochs: int,
          show_graph: bool = True):
    model.train()
    avg_losses = []

    for epoch in range(epochs):
        epoch_losses = []
        for image, target in tqdm(data_loader, desc=f"training on epoch {epoch} of {epochs}"):
            image = image.to(device)
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
            scores = model([image], [target])  # Somehow my GPU cannot handle more than a single image.
            total_loss = scores["loss_classifier"] + scores["loss_box_reg"] + \
                scores["loss_objectness"] + scores["loss_rpn_box_reg"]
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(total_loss.item())
        avg_losses.append(sum(epoch_losses) / len(epoch_losses))
        save(model, path="model-parameters")

    if show_graph:
        plt.plot(avg_losses)
        plt.title(f"Stats for Fast-R-CNN")
        plt.ylabel("Average Losses")
        plt.xlabel("Epochs")
        plt.show()


def target_transformation(label: dict):
    bboxes = []
    labels = []
    for annotation in label["annotation"]["object"]:
        bboxes.append([float(annotation["bndbox"]["xmin"]),
                       float(annotation["bndbox"]["ymin"]),
                       float(annotation["bndbox"]["xmax"]),
                       float(annotation["bndbox"]["ymax"])])
        labels.append(classes.index(annotation["name"]))
    return {"boxes": torch.tensor(bboxes), "labels": torch.tensor(labels)}


def main():
    config = Config("config.json")

    train_data = VOCDetection(root=config["voc-segmentation"],
                              year="2012",
                              image_set="train",
                              download=False,
                              transform=transforms.ToTensor(),
                              target_transform=target_transformation)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=None,
                              shuffle=True,
                              num_workers=4)

    # Define model:
    model = fasterrcnn_resnet50_fpn(weigths=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=22)
    try:
        load(model, path="model-parameters/FasterRCNN_only_last_layers.pt")
    except FileNotFoundError:
        print("Could not load model weights, continuing with default weights.")
    # Define what parameters are being updated:
    for parameter in model.parameters():
        parameter.requires_grad = False
    # for index, parameter in enumerate(model.parameters()):
    #     if index > 11:
    #         parameter.requires_grad = True
    for parameter in model.roi_heads.box_predictor.parameters():
        parameter.requires_grad = True
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        epochs=20,
        show_graph=True
    )


if __name__ == "__main__":
    main()
