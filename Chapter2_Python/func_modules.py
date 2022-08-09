import sys
# from sys import exit
# import tensorflow as tf
# import numpy as np


def print_dict_data(input_dict):
    for k, v in input_dict.items():
        print(k, v)


def main():
    my_dict = {"a": 1, "b": 2, "c": 3}
    print_dict_data(my_dict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
