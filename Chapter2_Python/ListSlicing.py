list_a = [i for i in range(10)]

print(list_a)

list_b = [list_a[i] for i in range(3)]

print(list_b)

# List Slicing

#              Start, Stop, Step
list_c = list_a[0:3:1]

print(list_c)

list_c[0] = -100

print(list_a)
print(list_c)
