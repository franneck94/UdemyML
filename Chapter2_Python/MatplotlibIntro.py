import matplotlib.pyplot as plt


grades_jan = [56, 64, 78, 100]
grades_ben = [86, 94, 98, 90]

# Plot
plt.plot(range(len(grades_jan)), grades_jan, color="blue")
plt.plot(range(len(grades_ben)), grades_ben, color="red")
plt.legend(["Jan", "Ben"])
plt.xlabel("Course")
plt.ylabel("Grade in %")
plt.title("Jan vs. Ben")
plt.show()

# Scatter
plt.scatter(range(len(grades_jan)), grades_jan, color="blue")
plt.scatter(range(len(grades_ben)), grades_ben, color="red")
plt.legend(["Jan", "Ben"])
plt.xlabel("Course")
plt.ylabel("Grade in %")
plt.title("Jan vs. Ben")
plt.show()
