import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# By Tianfeng LYU ----------------------------------------
# clean the data and perform pre-processing
def data_preprocessing(data_path,test_size):
	if 'banknote' in data_path:
		colname1 = ['variance','skewness','curtosis','entropy','class']
		data = get_data(data_path, None, colname1, None)
	elif 'kidney' in data_path:
		data = get_data(data_path, 0, None, 0)
		data = textclean(data)
		data = imputation(data)
	else:
		print('cannot handle this dataset')
	data = standardize(data)
	return mysplit(data, test_size)

# By Xueyao JI ----------------------------------------
# load dataset
def get_data(data_path,head,colname,index):
    # explanation:
    #    we process csv file only
    #    when run the program for the first time, transforme txt to csv
    if data_path.split('.')[-1] == 'txt':
        save_path = './data/'+data_path.split('/')[-1].split('.')[0]+'.csv'
        if os.path.exists(save_path)==False:
            data_txt = csv.reader(open(data_path, 'r'), delimiter = ',')
            data_csv = csv.writer(open(save_path, 'w', encoding = 'utf8'),  lineterminator = '\n')
            data_csv.writerows(data_txt)
        data = pd.read_csv(save_path, header = head, names = colname, index_col = index)
    elif data_path.split('.')[-1] == 'csv':
        data = pd.read_csv(data_path, header = head, names = colname, index_col = index)
    else:
        print('wrong file type')
    return data

# By Tianfeng LYU ----------------------------------------
# Processing string features
def textclean(data):
    data[['htn','dm','cad','pe','ane']] = data[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':0,'no':1})
    data['cad'] = data['cad'].replace(to_replace='\tno',value=0)
    data['dm'] = data['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
    data[['rbc','pc']] = data[['rbc','pc']].replace(to_replace={'abnormal':0,'normal':1})
    data[['pcc','ba']] = data[['pcc','ba']].replace(to_replace={'present':0,'notpresent':1})
    data[['appet']] = data[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
    #print("-------------------")
    #print(data.shape)
    data[['classification']] = data[['classification']].replace(to_replace={'ckd':0,'ckd\t':0,'notckd':1})
    #print("==================")
    for i in ['pcv','wc','rc']:
        data[i] = data[i].str.extract('(\d+)')
        data[i] = data[i].astype(float)
    return data

# By Tianfeng LYU ----------------------------------------
# replace missing values by average or median values
def imputation(data):
    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = data[column].mean()
        data[column].fillna(mean_val, inplace=True)
    return data

# By Tianfeng LYU ----------------------------------------
# center and normalize the data
def standardize(data):
    data = np.array(data)
    scaler = StandardScaler()
    scaler.fit(data[:,0:np.size(data,1)-1])
    return data

# By Tianfeng LYU ----------------------------------------
# split the dataset
def mysplit(data,test_size):
    X, y = data[:,0:np.size(data,1)-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


