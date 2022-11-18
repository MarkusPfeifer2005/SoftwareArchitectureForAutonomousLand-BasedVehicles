#!usr/bin/env python
import json
import os


class Config:
    """Just use like a dictionary."""
    file = "config.json"

    def __getitem__(self, item: str) -> object:
        """Automatically prompts to input path if none specified."""
        try:
            with open(self.file, 'r') as file:
                return json.load(file)[item]
        except FileNotFoundError or KeyError:
            value = input(f"Enter value for {item}: ")
            self.__setitem__(item, value)
            print("Value accepted.")
            return value

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
    if not os.path.isdir("model_parameters"):
        os.mkdir("model_parameters")


if __name__ == "__main__":
    main()
