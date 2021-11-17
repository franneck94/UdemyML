# == (Equal)
# < (Less than)
# > (Greater than)
# != (Not equal)
# <= (Less or equal than)
# >= (Greater or equal than)

i_am_broke = True

if i_am_broke:
    print("I am broke.")
else:
    print("I am not broke.")

my_bank_account = -10

if my_bank_account <= 0:
    print("I am broke.")
else:
    print("I am not broke.")

my_age = 10

if my_age < 18:
    print("You are a child")
elif my_age < 66:
    print("You are an adult")
else:
    print("You are a pensioner")
