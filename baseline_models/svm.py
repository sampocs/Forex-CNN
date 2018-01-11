import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np 
from sklearn import svm
from sklearn import metrics
from gen_train_set import gen_raw_class
import time

"""SVM Single Classifier for Price Movement"""

train, val, test = gen_raw_class()

#Separate sets
x_train, y_train = train
x_test, y_test = test

#Flatten
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

#Fit SVM
start_time = time.time()
clf = svm.SVC(cache_size=1000, class_weight='balanced', verbose=True)
clf.fit(x_train, y_train)
print "Time to train: {}".format(time.time() - start_time)

#Prediction
pred = clf.predict(x_test)

for a, p in zip(y_test, pred):
	print a, p

acc = metrics.accuracy_score(y_test, pred)
cohen = metrics.cohen_kappa_score(y_test, pred)
prec, rec, fscore, _ = metrics.precision_recall_fscore_support(y_test, pred)
print "Accuracy: {}, Cohen Kappa: {}\nPrecision: {}, Recall: {}, F-score: {}".format(acc, cohen, prec, rec, fscore)


