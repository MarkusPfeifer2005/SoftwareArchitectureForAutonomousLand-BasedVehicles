#!usr/bin/env python
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection

from init import Config
from object_detection import load, AnnotatedImage, target_transformation

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


def main():
    config = Config("config.json")

    train_data = VOCDetection(root=config["voc-segmentation"],
                              year="2012",
                              image_set="val",
                              download=False,
                              transform=transforms.ToTensor(),
                              target_transform=target_transformation)

    # Define model:
    model = fasterrcnn_resnet50_fpn(weigths=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=22)
    try:
        load(model, path="model-parameters/FasterRCNN.pt")
    except FileNotFoundError:
        print("Could not load model weights, continuing with default weights.")
    model.eval()

    for index, (image, target) in enumerate(train_data):
        scores = model([image])
        AnnotatedImage(image, scores).show()
        AnnotatedImage(image, [target]).show()
        # if input("press 'enter' to continue or 'q' to quit: ").lower() == 'q':
        if index > 7:
            break


if __name__ == "__main__":
    main()
