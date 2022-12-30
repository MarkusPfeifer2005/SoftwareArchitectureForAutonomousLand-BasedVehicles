#!/usr/bin/env python3.10

import os

import torch
from torch.utils.data import Dataset

from init import Config


class Names(Dataset):
    _file_extension = ".txt"
    _encoding = "utf-8"

    def __init__(self, root: str):
        """Reads whole dataset into memory!"""
        self._root = root
        self._languages = self._get_languages()
        self._number_languages = len(self._languages)
        self._characters = self._get_characters()
        self._number_characters = len(self._characters)

        # Prepare data.
        self._data = []
        for file in os.listdir(self._root):
            if file.endswith(self._file_extension):
                encoded_language = self._encode_language(language=file.replace(self._file_extension, ''))
                with open(os.path.join(self._root, file), 'r', encoding=self._encoding) as names_file:
                    for line in names_file:
                        for name in line:
                            encoded_name = []
                            for character in name:
                                encoded_name.append(self._encode_character(character))
                            self._data.append((encoded_name, encoded_language))

    def _encode_language(self, language: str) -> list[int]:
        assert language in self._languages
        encoding = [0 for _ in range(self._number_languages)]
        encoding[self._languages.index(language)] = 1
        return encoding

    def _encode_character(self, character: str) -> list[int]:
        assert character in self._characters
        encoding = [0 for _ in range(self._number_characters)]
        encoding[self._characters.index(character)] = 1
        return encoding

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, target = self._data[index]
        return torch.tensor(data), torch.tensor(target)

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


def main():
    config = Config("../../config.json")
    names = Names(root=config["names"])
    print(names[0])


if __name__ == "__main__":
    main()
