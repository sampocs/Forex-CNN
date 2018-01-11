import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np 
from sklearn import linear_model
from sklearn import metrics
from gen_train_set import gen_raw_reg

"""Linear Regression Next Price Prediction"""

train, val, test = gen_raw_reg()

#Separate sets
x_train, y_train = train
x_test, y_test = test

#Flatten
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

#Fit linear regression model
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

#Prediction
pred = reg.predict(x_test)

#Mean squared and R2 metrics
rmse = metrics.mean_squared_error(y_test, pred)
r2_score = metrics.r2_score(y_test, pred)

print "Means Squared Error: {}, R2 Score: {}".format(rmse, r2_score)

