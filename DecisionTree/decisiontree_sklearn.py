from __future__ import division
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
	iris = load_iris()
	X = iris.data
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	clf = DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print "score : %s" % score

	dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
	graph = graphviz.Source(dot_data)
	graph.render("iris")
