import csv
import numpy as np

with open('data/raw_candles.csv', 'rb') as csvfile:
    raw = list(csv.reader(csvfile, delimiter=','))

x = np.array( [ np.array([float(line[i]) for i in range(1, 5)]) for line in raw[1:] ] )

overfit = x[:(x.shape[0] % 1000), :]
x = x[(x.shape[0] % 1000):, :]

train = []
val = []
test = []
for i, points in enumerate(x):
	if i % 11 == 3:
		val.append(points)
	elif i % 11 == 7:
		test.append(points)
	else:
		train.append(points)

train = np.array(train)
val = np.array(val)
test = np.array(test)

np.savetxt("data/raw_train.csv", train, delimiter=",")
np.savetxt("data/raw_val.csv", val, delimiter=",")
np.savetxt("data/raw_test.csv", test, delimiter=",")
np.savetxt("data/overfit.csv", overfit, delimiter=",")
