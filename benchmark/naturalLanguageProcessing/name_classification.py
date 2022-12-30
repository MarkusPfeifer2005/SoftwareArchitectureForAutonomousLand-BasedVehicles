#!/usr/bin/env python3.10

import os

import torch

from init import Config


class Names:
    _file_extension = ".txt"
    _encoding = "utf-8"

    def __init__(self, root: str):
        self._root = root

    def get_languages(self) -> list[str]:
        languages = set()
        for file in os.listdir(self._root):
            if file.endswith(self._file_extension):
                languages.add(file.replace(self._file_extension, ''))
        return sorted(languages)

    def get_number_languages(self) -> int:
        return len(self.get_languages())

    def get_used_characters(self) -> list[str]:
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
    print(names.get_languages())
    print(names.get_used_characters())


if __name__ == "__main__":
    main()
