from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from random import shuffle


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
		

if __name__ == "__main__":
	iris = load_iris()
	X = iris.data[:100, :2]
	y = iris.target[:100]
	y[y == 0] = -1
	xlabel = iris.feature_names[0]
	ylabel = iris.feature_names[1]

	X_0 = X[:50]
	X_1 = X[50:]

	plt.figure("iris")
	plt.scatter(X_0[:, 0], X_0[:, 1], label = '-1')
	plt.scatter(X_1[:, 0], X_1[:, 1], label = '1')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()

	clf = Perceptron(X, y)
	clf.fit()
	
	X_simul = np.arange(4, 8)
	y_simul = -(clf.w[0] * X_simul + clf.b)/clf.w[1]
	plt.plot(X_simul, y_simul, color = 'red', label = 'model')
	plt.legend()
	plt.show()
