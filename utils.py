import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score
# from keras import Sequential
#import matplotlib.pyplot as plt


# clean the data and perform pre-processing
def get_data(data_path, test_size=0.2):

	# load dataset
	# explanation:
	#    we process csv file only
	#    when run the program for the first time, transforme txt to csv
	if data_path.split('.')[-1] == 'txt':
		save_path = './data/'+data_path.split('/')[-1].split('.')[0]+'.csv'
		if os.path.exists(save_path)==False:
			data_txt = csv.reader(open(data_path, "r"), delimiter = ',')
			data_csv = csv.writer(open(save_path, 'w', encoding = 'utf8'),  lineterminator = '\n')
			data_csv.writerows(data_txt)
		data = pd.read_csv(save_path)
	elif data_path.split('.')[-1] == 'csv':
		data = pd.read_csv(data_path)
	else:
		print('wrong file type')

	# replace missing values by average or median values
	### todo ###

	# center and normalize the data
	### todo ###
	
	# split the dataset
	# 1) -  between training set and test set
	data = np.array(data)
	#X = data[["variance","skewness","curtosis","entropy"]]
	#y = data["Target"]
	X, y = data[:,:4], data[:,-1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
	#print(X_train.shape)
	return X_train, X_test, y_train, y_test

	# 2) - split the training set for cross-validation


# models
# pca

data_path = "./data/data_banknote_authentication.txt"
# data_path = "./data/kidney_disease.csv"
get_data(data_path)