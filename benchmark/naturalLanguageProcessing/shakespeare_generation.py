#!/usr/bin/env python3.10

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from init import Config


device = "cuda" if torch.cuda.is_available() else "cpu"  # Global.


class ShakespeareData:
    def __init__(self, root: str, sequence_length: int):
        self.sequence_length = sequence_length
        self._root = root
        self._characters = self.get_characters()
        self.number_characters = len(self._characters)

    def __iter__(self) -> tuple:
        sequence = []
        with open(self._root, 'r') as file:
            for line in file:
                for character in line:
                    sequence.append(self._encode_character(character))
                    if len(sequence) == self.sequence_length + 1:  # +1 due to added target.
                        data = sequence[:-1]
                        target = sequence[-1]
                        yield torch.tensor(data).unsqueeze(1), torch.tensor([target.index(1.)])
                        sequence.pop(0)

    def get_characters(self) -> list[str]:
        used_characters: set = set()
        with open(self._root, 'r') as shakespeare_works:
            for line in shakespeare_works:
                for character in line:
                    used_characters.add(character)
        return sorted(used_characters)

    def _encode_character(self, character: str) -> list[float]:
        assert character in self._characters
        encoding = [0. for _ in range(self.number_characters)]
        encoding[self._characters.index(character)] = 1.
        return encoding


class ShakespeareGenerator(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ShakespeareGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(size=(1, sequence.size(1), self.hidden_size)).to(device)
        sequence_predictions, hidden_n = self.rnn(sequence, hidden)
        return self.linear(sequence_predictions[-1])

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print("Model saved successfully.")

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        print("Model loaded successfully.")


def train(dataset: ShakespeareData, model, epochs: int, criterion, optimizer):
    model.train()
    average_losses = []
    for _ in tqdm(range(epochs)):
        epoch_losses = []
        for data, target in dataset:
            data, target = data.to(device), target.to(device)
            scores = model.forward(data)
            loss = criterion(scores, target)
            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_losses.append(sum(epoch_losses) / len(epoch_losses))

    plt.plot(average_losses)
    plt.title(f"Stats for Model:")
    plt.ylabel("Average Losses")
    plt.xlabel("Epochs")
    plt.show()


def main():
    config = Config("../../config.json")

    shakespeare_data = ShakespeareData(root=config["shakespeare"], sequence_length=20)
    epochs = 4

    model = ShakespeareGenerator(input_size=shakespeare_data.number_characters,
                                 hidden_size=70,
                                 output_size=shakespeare_data.number_characters).to(device)

    try:
        model.load("../../model-parameters/shakespeare_generator")
    except FileNotFoundError:
        pass
    train(dataset=shakespeare_data,
          model=model,
          epochs=epochs,
          criterion=torch.nn.CrossEntropyLoss(),
          optimizer=torch.optim.SGD(model.parameters(), lr=0.005))
    model.save("../../model-parameters/shakespeare_generator")


if __name__ == "__main__":
    main()
