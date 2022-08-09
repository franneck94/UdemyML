class CarData:
    def __init__(self, name, oem, hp, year):
        self.name = name
        self.oem = oem
        self.hp = hp
        self.year = year

    def print_data(self):
        print("Name: " + str(self.name))
        print("OEM: " + str(self.oem))
        print("HP: " + str(self.hp))
        print("Year: " + str(self.year))


class Audi(CarData):
    def __init__(self, name, hp, year):
        super().__init__(name, "Audi", hp, year)


class Mercedes(CarData):
    def __init__(self, name, hp, year):
        super().__init__(name, "Mercedes", hp, year)


def main():
    car1 = Audi("RS3", 400, 2023)
    car1.print_data()

    car2 = Mercedes("A45S", 421, 2023)
    car2.print_data()


if __name__ == "__main__":
    main()
