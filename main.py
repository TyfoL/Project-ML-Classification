import os
import time
from utils import get_data
#from network import network
from methods import logistic_regression

if __name__ == "__main__":
	
	# load data
	data_path_1 = "./data/data_banknote_authentication.txt" # print(data.shape) (1371,5)
	data_path_2 = "./data/kidney_disease.csv" # print(data.shape) (400,26)
	data_path = data_path_1

	X_train, X_test, y_train, y_test = get_data(data_path, test_size=0.2)

	# choose from logistic_regression
	method = "logistic_regression"

	start = time.time()
	if method == "logistic_regression":
		if "banknote" in data_path:
			accuracy = logistic_regression(data_path, num_gradient_descent=3000, rate=0.002) # txt file
			print("accuracy of logistic regression is ",accuracy)
		else:
			print("logistic_regression is not suitable for current dataset!")
	end = time.time()
	print("running time is ", (end-start), "s")
 