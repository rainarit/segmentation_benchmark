import pickle

file = open('hist.txt', 'rb')
hist = pickle.load(file)

print(hist)