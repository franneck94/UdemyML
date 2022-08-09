# append: adds element to a list
# pop: removes element from a list

# list comprehension
l = [i**2 for i in range(1, 6)]
m = [i for i in range(1, 6)]

for i in range(len(l)):
    print(i, l[i])

# enumerate
for i, val in enumerate(l):
    print(i, val)

for i in range(len(l)):
    print(l[i], m[i])

# zip
for val1, val2 in zip(l, m):
    print(val1, val2)
