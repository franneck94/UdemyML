my_list = [12, "12", True, None]

for element in my_list:
    print(element)

for i in range(len(my_list)):  # [0, 1, 2, 3]
    print(my_list[i])

# key: value
my_dict = {"a": 1, "b": 2, "c": 3}

print(my_dict["a"])

for v in my_dict:  # keys
    print(v)

for k, v in my_dict.items():  # keys, values
    print(k, v)

for v in my_dict.values():  # values
    print(v)


i = 0
while True:
    print("Hello")
    i += 1
    if i >= 10:
        break  # exit the loop
