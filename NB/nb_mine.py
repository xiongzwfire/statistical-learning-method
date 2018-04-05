from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class GaussianNB:
	def __init__(self, X_train, y_train, n_neighbors = 5, p = 2):
		self.X_train = X_train
		self.y_train = y_train
		self.get_params()

	def get_params(self):
		samples_num = len(self.y_train)
		label_count = Counter(self.y_train).items()
		self.class_ = np.array([_[0] for _ in label_count])
		self.class_count_ = np.array([_[1] for _ in label_count])
		self.class_prior_ = self.class_count_/samples_num

		sample_split = [self.X_train[self.y_train == _] for _ in self.class_]
		self.theta_ = np.array([np.transpose(_).mean(axis = 1) for _ in sample_split])
		self.sigma_ = np.array([np.transpose(_).var(axis = 1) for _ in sample_split])

	def gaussian_func(self, X):
		prob_list = []
		for theta, sigma, prior in zip(self.theta_, self.sigma_, self.class_prior_):
			prob_ = np.prod(1/np.sqrt(2 * np.pi * sigma) * np.exp(-np.square(X - theta)/(2 * sigma))) * prior
			prob_list.append(prob_)
		return np.array(prob_list)

	def predict(self, X_test):
		label_list = []
		for X in X_test:
			prob_list = self.gaussian_func(X)
			label = self.class_[np.argmax(prob_list)]
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

	plt.figure("nb-mine")
	plt.scatter(X_0[:, 0], X_0[:, 1], label = '-1')
	plt.scatter(X_1[:, 0], X_1[:, 1], label = '1')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()

	clf = GaussianNB(X_train, y_train)
	score = clf.score(X_test, y_test)
	print "score : %s" % score
	
	y_pre = clf.predict(X_test)
	X_test_pre_0 = X_test[y_pre == -1]
	X_test_pre_1 = X_test[y_pre == 1]
	plt.scatter(X_test_pre_0[:, 0], X_test_pre_0[:, 1], color = 'r', label = 'pre -1')
	plt.scatter(X_test_pre_1[:, 0], X_test_pre_1[:, 1], color = 'k', label = 'pre 1')
	plt.legend()
	plt.show()
