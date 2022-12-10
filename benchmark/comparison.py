#!/usr/bin/env python
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from nearest_neighbour import Cifar10Dataset
from nearest_neighbour import evaluate
from mlib.scratch import SVMLossVectorized, StochasticGradientDecent
from linear_classification import LinearClassifier, ExperimentalModel, SigmoidModel
from torch_classification import TorchLinearClassifier, TorchExperimentalModel, TorchSigmoidModel
from init import Config

import torch
import torch.nn as nn


def train(models: list, model_class: str, dataset: Cifar10Dataset, criteria: list, optimizers: list, epochs: int,
          completed_epochs: int = 0, normalize: bool = False):
    avg_losses0, avg_losses1 = [], []
    for _ in tqdm(range(completed_epochs, completed_epochs + epochs), desc=f"Training the {model_class} models"):
        batch_losses0, batch_losses1 = [], []
        for batch in dataset:
            data = batch[b"data"].astype("float64") / 255 if normalize else batch[b"data"].astype("float64")
            targets = batch[b"labels"]

            m_scores = models[0].forward(data)
            t_scores = models[1].forward(torch.from_numpy(data))
            assert np.round(m_scores, 3).tolist() == torch.round(t_scores, decimals=3).tolist(), "scores are unequal"

            m_loss = criteria[0](m_scores, targets)
            t_loss = criteria[1](t_scores, torch.tensor(targets))
            assert round(m_loss, 3) == round(t_loss.item(), 3), f"{m_loss} != {t_loss.item()}"

            m_grads = []
            grad = criteria[0].backward()
            optimizers[0].step(grad=grad)  # Should work?
            for layer in reversed(models[0].layers):
                grad, parameter_grads = layer.backward(grad)
                m_grads += reversed(parameter_grads)
            m_grads.reverse()

            t_loss.backward()
            t_grads = []
            for param in models[1].parameters():
                t_grads.append(param.grad.numpy())
            optimizers[1].step()
            optimizers[1].zero_grad()

            # The following assertion does not work due to torch transposing weights!
            # assert [np.round(g, 6).tolist() for g in m_grads] == [np.round(g, 6).tolist() for g in t_grads]

            batch_losses0.append(m_loss)
            batch_losses1.append(t_loss.item())
        avg_losses0.append(sum(batch_losses0) / len(batch_losses0))
        avg_losses1.append(sum(batch_losses1) / len(batch_losses1))

    plt.title(label=model_class)
    plt.plot(avg_losses0)
    plt.plot(avg_losses1)
    plt.ylabel("Average Losses")
    plt.xlabel("Epochs")
    plt.show()


def evaluate_torch_cifar(model,
              dataset: Cifar10Dataset,
              normalize: bool = False,
              images: slice = slice(None, None, None)) -> float:
    """For Cifar10 only."""
    total = correct = 0
    for batch in dataset:
        for img, lbl in zip(batch[b"data"][images], batch[b"labels"][images]):
            img = img.astype("float64") / 255 if normalize else img.astype("float64")
            img = torch.from_numpy(img)
            target = dataset.labels[lbl].decode()
            scores = model(img).argmax().item()
            prediction = dataset.labels[scores].decode()

            total += 1
            if prediction == target:
                correct += 1

    print(f"{correct} of {total} examples were correct resulting in an accuracy of {correct/total*100:.2f}%.")
    return correct/total*100

def main():
    config = Config("../config.json")
    train_set: Cifar10Dataset = Cifar10Dataset(batches=slice(0, 1), root=config["cifar"])
    evaluation_set: Cifar10Dataset = Cifar10Dataset(batches=slice(4, 5), root=config["cifar"])
    epochs = int(100)

    # define models
    models = {
        "linear": (LinearClassifier(), TorchLinearClassifier()),
        "experimental": (ExperimentalModel(), TorchExperimentalModel()),
        "sigmoid": (SigmoidModel(), TorchSigmoidModel().double())  # parameters with datatype double
              }

    for key in models:
        my_model = models[key][0]
        torch_model = models[key][1]

        # equalize parameters
        torch_model.linear1.weight.data = torch.from_numpy(my_model.layers[0].parameters[0].T.copy())
        torch_model.linear1.bias.data = torch.from_numpy(my_model.layers[0].parameters[1].copy())
        if len(my_model.layers) >= 2:
            torch_model.linear2.weight.data = torch.from_numpy(my_model.layers[1].parameters[0].T.copy())
            torch_model.linear2.bias.data = torch.from_numpy(my_model.layers[1].parameters[1].copy())

        # define criteria
        my_criterion = SVMLossVectorized()
        torch_criterion = nn.MultiMarginLoss()

        # define optimizers
        my_optimizer = StochasticGradientDecent(model_layers=my_model.layers, lr=.001, momentum=.5)
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=.001, momentum=.5)

        # train the models
        train(models=[my_model, torch_model], dataset=train_set, criteria=[my_criterion, torch_criterion],
              optimizers=[my_optimizer, torch_optimizer], epochs=epochs, model_class=key)

        # evaluate_models:
        evaluate(my_model, train_set)
        evaluate(my_model, evaluation_set)
        evaluate_torch_cifar(torch_model, train_set)
        evaluate_torch_cifar(torch_model, evaluation_set)


if __name__ == "__main__":
    main()
