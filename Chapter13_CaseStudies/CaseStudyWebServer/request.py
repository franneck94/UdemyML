import requests

# URL
url = 'http://localhost:5000/api'

# Change the value of experience that you want to test
r = requests.post(url,json={'x': [1, 0, 0, 0, 40, 10, 10]})
print(r.json())