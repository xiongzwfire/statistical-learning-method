from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron


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

	clf = Perceptron(max_iter = 1000)
	clf.fit(X, y)

	
	
	X_simul = np.arange(4, 8)
	y_simul = -(clf.coef_[0, 0] * X_simul + clf.intercept_)/clf.coef_[0, 1]
	plt.plot(X_simul, y_simul, color = 'red', label = 'model')
	plt.legend()
	plt.show()
