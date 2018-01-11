import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from gen_train_set import gen_raw_class
import time

"""Random Forest Classification for Price Movement"""

train, val, test = gen_raw_class()

#Separate sets
x_train, y_train = train
x_test, y_test = test

#Flatten
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

#Fit SVM
min_samples_leaf = 20
start_time = time.time()
clf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, verbose=1)
clf.fit(x_train, y_train)
print "Time to train: {}".format(time.time() - start_time)

#Prediction
pred = clf.predict(x_test)

#Metrics
acc = metrics.accuracy_score(y_test, pred)
cohen = metrics.cohen_kappa_score(y_test, pred)
prec, rec, fscore, _ = metrics.precision_recall_fscore_support(y_test, pred)
print "Accuracy: {}, Cohen Kappa: {}\nPrecision: {}, Recall: {}, F-score: {}".format(acc, cohen, prec, rec, fscore)