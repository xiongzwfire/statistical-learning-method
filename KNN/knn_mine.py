from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class KNeighborsClassifier:

	def __init__(self, X_train, y_train, n_neighbors = 5, p = 2):
		self.X_train = X_train
		self.y_train = y_train
		self.n_neighbors = n_neighbors
		self.p = p

	def cal_dist(self, X1, X2):
		dist = np.linalg.norm(X1-X2, ord = self.p)	
		return dist

	def predict(self, X_test):
		label_list = []
		for X in X_test:
			dist_list = []
			for X_i, y_i in zip(self.X_train, self.y_train):
				dist = self.cal_dist(X, X_i)
				dist_list.append((dist, y_i))
			dist_list.sort()
			knn_list = dist_list[: self.n_neighbors]
			label = Counter([_[1] for _ in knn_list]).most_common(1)[0][0]
			label_list.append(label)
		return np.array(label_list)

	def score(self, X_test, y_test):
		total_num = len(X_test)
		pre = (self.predict(X_test) == y_test).sum()
		score = pre/total_num
		return score
		

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

	plt.figure("knn-mine")
	plt.scatter(X_0[:, 0], X_0[:, 1], label = '-1')
	plt.scatter(X_1[:, 0], X_1[:, 1], label = '1')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()

	clf = KNeighborsClassifier(X_train, y_train)
	score = clf.score(X_test, y_test)
	print "score : %s" % score
	
	y_pre = clf.predict(X_test)
	X_test_pre_0 = X_test[y_pre == -1]
	X_test_pre_1 = X_test[y_pre == 1]
	plt.scatter(X_test_pre_0[:, 0], X_test_pre_0[:, 1], color = 'r', label = 'pre -1')
	plt.scatter(X_test_pre_1[:, 0], X_test_pre_1[:, 1], color = 'k', label = 'pre 1')
	plt.legend()
	plt.show()
