import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from gen_train_set import gen_raw_reg
import time

"""Random Forest Regression Next Price Prediction"""

train, val, test = gen_raw_reg()

#Separate sets
x_train, y_train = train
x_test, y_test = test

#Flatten
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

#Fit linear regression model
min_samples_leaf = 2
start_time = time.time()
reg = RandomForestRegressor(verbose=1, min_samples_leaf=min_samples_leaf)
reg.fit(x_train, y_train)
print "Time to train: {}".format(time.time() - start_time)

#Prediction
pred = reg.predict(x_test)

#Mean squared and R2 metrics
rmse = metrics.mean_squared_error(y_test, pred)
r2_score = metrics.r2_score(y_test, pred)

print "Means Squared Error: {}, R2 Score: {}".format(rmse, r2_score)

