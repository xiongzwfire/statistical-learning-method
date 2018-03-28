from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from random import shuffle
from sklearn.model_selection import train_test_split


class Perceptron:
	def __init__(self, X_train, y_train, lr = 0.1):
		self.X_train = X_train
		self.y_train = y_train
		self.w = np.zeros(self.X_train.shape[1])
		self.b = 0
		self.lr = lr

	def fit(self):
		iter_flag = True
		while iter_flag:
			train_set = zip(self.X_train, self.y_train)
			shuffle(train_set)
			for X_i, y_i in train_set:
				if y_i * (np.dot(self.w, X_i) + self.b) <= 0:
					self.w += self.lr * np.dot(X_i, y_i)
					self.b += self.lr * y_i
					break
			else:
				iter_flag = False

		print "Perceptron OK!"

	def predict(self, X_test):
		label_list = []
		for X in X_test:
			label = np.sign(np.dot(self.w, X) + self.b)
			label = label if label else -1
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

	plt.figure("perceptron-mine")
	plt.scatter(X_0[:, 0], X_0[:, 1], label = '-1')
	plt.scatter(X_1[:, 0], X_1[:, 1], label = '1')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()

	clf = Perceptron(X, y)
	clf.fit()
	score = clf.score(X_test, y_test)
	print "score : %s" % score
	
	y_pre = clf.predict(X_test)
	X_test_pre_0 = X_test[y_pre == -1]
	X_test_pre_1 = X_test[y_pre == 1]
	plt.scatter(X_test_pre_0[:, 0], X_test_pre_0[:, 1], color = 'r', label = 'pre -1')
	plt.scatter(X_test_pre_1[:, 0], X_test_pre_1[:, 1], color = 'k', label = 'pre 1')
	plt.legend()

	
	X_simul = np.arange(4, 8)
	y_simul = -(clf.w[0] * X_simul + clf.b)/clf.w[1]
	plt.plot(X_simul, y_simul, color = 'g', label = 'model')
	plt.legend()
	plt.show()
