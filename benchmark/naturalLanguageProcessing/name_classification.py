#!/usr/bin/env python3.10

import os
from random import shuffle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from init import Config


class Names(Dataset):
    _file_extension = ".txt"
    _encoding = "utf-8"

    def __init__(self, root: str, random: bool = True):
        """Reads whole dataset into memory!"""
        self._root = root
        self._languages = self._get_languages()
        self.number_languages = len(self._languages)
        self._characters = self._get_characters()
        self.number_characters = len(self._characters)
        self._random = random

        # Prepare data.
        self._data = []
        for file in os.listdir(self._root):
            if file.endswith(self._file_extension):
                encoded_language = self._encode_language(language=file.replace(self._file_extension, ''))
                with open(os.path.join(self._root, file), 'r', encoding=self._encoding) as names_file:
                    for name in names_file:
                        encoded_name = []
                        for character in name:
                            encoded_name.append(self._encode_character(character))
                        self._data.append((encoded_name, encoded_language))
        self.randomize()

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        for data, target in self._data:
            yield torch.tensor(data).unsqueeze(1), torch.tensor(target)

    def _encode_language(self, language: str) -> list[float]:
        assert language in self._languages
        encoding = [0. for _ in range(self.number_languages)]
        encoding[self._languages.index(language)] = 1.
        return encoding

    def _encode_character(self, character: str) -> list[float]:
        assert character in self._characters
        encoding = [0. for _ in range(self.number_characters)]
        encoding[self._characters.index(character)] = 1.
        return encoding

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, target = self._data[index]
        return torch.tensor(data).unsqueeze(1), torch.tensor(target)

    def __len__(self) -> int:
        length = 0
        for file in os.listdir(self._root):
            if file.endswith(self._file_extension):
                with open(os.path.join(self._root, file), 'r', encoding=self._encoding) as names_file:
                    length += len(names_file.readlines())
                length += len(file)
        return length

    def _get_languages(self) -> list[str]:
        languages = set()
        for file in os.listdir(self._root):
            if file.endswith(self._file_extension):
                languages.add(file.replace(self._file_extension, ''))
        return sorted(languages)

    def _get_characters(self) -> list[str]:
        used_characters = set()
        for file in os.listdir(self._root):
            if file.endswith(self._file_extension):
                with open(os.path.join(self._root, file), 'r', encoding=self._encoding) as names_file:
                    for line in names_file:
                        for name in line:
                            for character in name:
                                used_characters.add(character)
        return sorted(used_characters)

    def randomize(self):
        if self._random:
            shuffle(self._data)


class MyRNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MyRNN, self).__init__()

        self._hidden = torch.zeros(size=(hidden_size, ))
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear3 = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
        self.nonlinearity = torch.nn.Tanh()

    def forward(self, name: torch.Tensor) -> torch.Tensor:
        """Takes an entire name."""
        self._hidden = torch.zeros_like(self._hidden)  # Reinitialize hidden for new sequence.

        for character in name:
            self._hidden = self.nonlinearity(self.linear1(character) + self.linear2(self._hidden))

        scores = self.linear3(self._hidden)
        return scores


def train(data: Names, model, epochs: int):
    data.randomize()
    for _ in tqdm(range(epochs)):
        for data, target in data:
            scores = model(data)
            print(data.shape, target.shape, scores.shape)


def main():
    config = Config("../../config.json")
    data = Names(root=config["names"])
    rnn = MyRNN(input_size=data.number_characters, hidden_size=50, output_size=data.number_languages)
    train(data=data, model=rnn, epochs=1)


if __name__ == "__main__":
    main()
