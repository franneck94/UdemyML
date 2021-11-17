# C++, Java: this


class Animal:
    def __init__(self, weight, height):
        self.weight = weight
        self.height = height

    def jump(self):
        print("Jump!")


def main():
    dog = Animal(10, 0.8)
    print(dog.height)
    print(dog.weight)
    dog.jump()

    cat = Animal(3, 0.3)
    print(cat.height)
    print(cat.weight)
    cat.jump()


if __name__ == "__main__":
    main()
