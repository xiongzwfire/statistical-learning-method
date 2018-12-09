from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
	iris = load_iris()
	X = iris.data[:100, :2]
	y = iris.target[:100]
	y[y == 0] = -1
	xlabel = iris.feature_names[0]
	ylabel = iris.feature_names[1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	X_0 = X_train[y_train == -1]
	X_1 = X_train[y_train == 1]

	plt.figure("lr-sklearn")
	plt.scatter(X_0[:, 0], X_0[:, 1], color = 'y', label = '-1')
	plt.scatter(X_1[:, 0], X_1[:, 1], color = 'b', label = '1')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()

	clf = LogisticRegression()
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print "score : %s" % score
	
	y_pre = clf.predict(X_test)
	X_test_pre_0 = X_test[y_pre == -1]
	X_test_pre_1 = X_test[y_pre == 1]
	plt.scatter(X_test_pre_0[:, 0], X_test_pre_0[:, 1], color = 'r', label = 'pre -1')
	plt.scatter(X_test_pre_1[:, 0], X_test_pre_1[:, 1], color = 'k', label = 'pre 1')
	plt.legend()
	plt.show()
