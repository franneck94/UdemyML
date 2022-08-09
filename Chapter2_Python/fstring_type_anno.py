class CarData:
    def __init__(self, name: str, oem: str, hp: int, year: int) -> None:
        self.name = name
        self.oem = oem
        self.hp = hp
        self.year = year

    def get_info(self) -> str:
        return f"Name: {self.name}, OEM: {self.oem}, HP: {self.hp}, Year: {self.year}"


class Audi(CarData):
    def __init__(self, name: str, hp: int, year: int) -> None:
        super().__init__(name, "Audi", hp, year)


class Mercedes(CarData):
    def __init__(self, name: str, hp: int, year: int) -> None:
        super().__init__(name, "Mercedes", hp, year)


def main():
    car1 = Audi("RS3", 400, 2023)
    info1 = car1.get_info()
    print(info1)

    car2 = Mercedes("A45S", 421, 2023)
    info2 = car2.get_info()
    print(info2)


if __name__ == "__main__":
    main()
