#!usr/bin/env python
import json
import os


class Config:
    """Just use like a dictionary."""

    def __init__(self, root: str):
        self.file = root

    def __getitem__(self, item: str) -> str:
        """Automatically prompts to input path if none specified."""
        with open(self.file, 'r') as file:
            try:
                return str(json.load(file)[item])
            except (FileNotFoundError, KeyError):
                value = input(f"Enter value for {item}: ")
                self.__setitem__(item, value)
                print("Value accepted.")
                return str(value)

    def __setitem__(self, key: str, value: object):
        try:
            with open(self.file, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}
        data[key] = value
        with open(self.file, 'w') as file:
            json.dump(data, file)


def main():
    config = Config("config.json")
    if not os.path.isdir(config["model-parameters"]):
        os.mkdir(config["model-parameters"])


if __name__ == "__main__":
    main()
