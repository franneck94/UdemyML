from typing import List


def list_max(input_list: List):
    max_value = input_list[0]

    for i in range(1, len(input_list)):
        if input_list[i] > max_value:
            max_value = input_list[i]

    print(max_value)


def list_min(input_list: List):
    max_value = input_list[0]

    for i in range(1, len(input_list)):
        if input_list[i] < max_value:
            max_value = input_list[i]

    print(max_value)


def main():
    list1 = [-2, 1, 2, -10, 22, -10]
    list_max(list1)
    list_min(list1)
    list2 = [-20, 123, 112, -10, 22, -120]
    list_max(list2)
    list_min(list2)


if __name__ == "__main__":
    main()
