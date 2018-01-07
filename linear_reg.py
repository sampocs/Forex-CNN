import numpy as np 
from sklearn import linear_model
from sklearn import metrics
from gen_train_set import gen_raw_reg, raw_mu, raw_var

train, val, test = gen_raw_reg()

#Separate sets
x_train, y_train = train
x_train = np.array(x_train)
x_test, y_test = test
x_test = np.array(x_test)

#De-normalize and flatten
x_train *= raw_var ** (1/2.0)
x_train += raw_mu
x_train = np.reshape(x_train, (x_train.shape[0], -1))

x_test*= raw_var ** (1/2.0)
x_test += raw_mu
x_test = np.reshape(x_test, (x_test.shape[0], -1))

#Fit linear regression model
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

pred = reg.predict(x_test)
y_test = np.array(y_test)

#Mean squared and R2 metrics
rmse = metrics.mean_squared_error(y_test, pred)
r2_score = metrics.r2_score(y_test, pred)
