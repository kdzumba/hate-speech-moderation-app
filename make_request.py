import requests

url = 'https://quasi-star.herokuapp.com/'

r = requests.post(url, json={'text': 'I hate food'})
print(r.json)