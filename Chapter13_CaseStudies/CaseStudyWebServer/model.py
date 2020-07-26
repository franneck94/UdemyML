import numpy as np
import pandas as pd
dataset = pd.read_csv('employee_data.csv')
dataset = dataset.drop("Unnamed: 0", 1)
dataset = dataset.drop("id", 1)
print(dataset.head())

dataset = dataset.as_matrix()
print(dataset)

x = dataset[:, :-1]
y = dataset[:, -1]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", dtype=np.int8)
enc.fit(x[:,[0]])
x_cat = enc.transform(x[:,[0]]).toarray()
print(x_cat.shape)
print(x_cat[0])

x_cat = np.append(x_cat, x[:,1:], axis=1)
print(x_cat.shape)
print(x_cat[0])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_cat, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
print("Score: ", score)

import pickle
pickle.dump(regr, open('model.pkl','wb'))
regr = pickle.load( open('model.pkl','rb'))
sample = np.array([1, 0, 0, 0, 40, 10, 10])
print(regr.predict([sample]))