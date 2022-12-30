#!/usr/bin/env python3.10

import torch

from init import Config


class Shakespeare:
    def __init__(self, root: str):
        self._root = root

    def __iter__(self):
        ...

    def get_used_characters(self) -> list[str]:
        used_characters: set = set()
        with open(self._root, 'r') as shakespeare_works:
            for line in shakespeare_works:
                for character in line:
                    used_characters.add(character)
        return sorted(used_characters)

    def get_num_used_characters(self) -> int:
        return len(self.get_used_characters())


def main():
    config = Config("../../config.json")
    shakespeare_text = Shakespeare(root=config["shakespeare"])
    print(shakespeare_text.get_used_characters())


if __name__ == "__main__":
    main()
