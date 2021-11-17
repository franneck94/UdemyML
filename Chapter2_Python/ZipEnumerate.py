list_a = [100.0, 200.0, -10.0]
list_b = [False, False, True]

# index
for idx in range(len(list_a)):
    print(idx, list_a[idx], list_b[idx])


print("")


# values for multiple iterables
for val_a, val_b in zip(list_a, list_b):
    print(val_a, val_b)


print("")


# index and value
for idx, val in enumerate(list_a):
    print(idx, val)


print("")

# index and values for multiple iterables
for idx, (val_a, val_b) in enumerate(zip(list_a, list_b)):
    print(idx, val_a, val_b)
