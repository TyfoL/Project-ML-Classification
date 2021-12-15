import os
import time
from utils import data_preprocessing
from methods import logistic_regression, Svm_acc, DecisionTree_acc, KNN_acc
from network import network

# By Xueyao JI ----------------------------------------
if __name__ == "__main__":

	# load data
	data_path_1 = "./data/data_banknote_authentication.txt"
	data_path_2 = "./data/kidney_disease.csv"
	
	data_path = data_path_2
	X_train, X_test, y_train, y_test = data_preprocessing(data_path, test_size=0.2)

	# choose from logistic_regression, network, Svm_acc, DecisionTree_acc, KNN_acc
	method = "network"

	start = time.time()

	if method == "logistic_regression":
		if "banknote" in data_path:
			print("using method logistic regression")
			accuracy = logistic_regression(data_path, num_gradient_descent=4000, rate=0.01) # txt file
			print("accuracy of logistic regression is ",accuracy)
		else:
			print("logistic_regression is not suitable for current dataset!")
	
	elif method == "network":
		print("using method network")
		network([X_train, X_test, y_train, y_test], net="deeper", batch_size=20, n_epochs=500, lr=0.05)
	
	elif method == "SVM_acc":
		print("using method SVM")
		Svm_acc(X_train, X_test, y_train, y_test)

	elif method == "DecisionTree_acc":
		print("using method DecisionTree")
		DecisionTree_acc(X_train, X_test, y_train, y_test)

	elif method == "KNN_acc":
		print("using method KNN")
		KNN_acc(X_train, X_test, y_train, y_test, n_neighbors=10)
	
	end = time.time()
	print("running time is ", (end-start), "s")
 