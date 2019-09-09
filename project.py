#necessary libraries imported
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv
import matplotlib.pyplot as plt

#reading from train.csv
data = np.array([])
with open('../Data/train.csv','r') as train_set:
    reader = csv.reader(train_set)
    for row in reader:
        if (row[0] != "X1"):
            for i in row:
                i = float(i)
                data = np.append(data, i)
#divide the data to x and y
sonuc = np.array([])
x = np.array([])
for i in range(len(data)):
    if(i == 595):
        sonuc = np.append(sonuc,data[i])
    elif((i+1)%596 == 0 and i!=0):
        sonuc = np.append(sonuc, data[i])
    else:
        x = np.append(x,data[i])
#reshaping the array x to 2 dimention
x = np.reshape(x,(120,595))
#Creating a pandas Dataframe amd put trainset in it
Dataset = pd.DataFrame()

for i in range(595):
    ColumnData = pd.DataFrame({'Column': x[:,i]})
    Dataset= pd.concat([Dataset,ColumnData],axis=1)

#Feature extraction PCA method applied results x_pca
scaler = StandardScaler()
scaler.fit(Dataset)
scaled_data = scaler.transform(Dataset)
pca = PCA(n_components=1)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

#reading from test.csv
test_x = np.array([])
y_pred = np.array([])

with open('../Data/test.csv','r') as train_set:
    reader = csv.reader(train_set)
    for row in reader:
        if (row[0] != "X1"):
            for i in row:
                i = float(i)
                test_x = np.append(test_x, i)
test_x = np.reshape(test_x,(80,595))

#passing the data to pandas dataframe
Testset = pd.DataFrame()
for i in range(595):
    ColumnData = pd.DataFrame({'Column': test_x[:,i]})
    Testset= pd.concat([Testset,ColumnData],axis=1)

#Feature selection to Test Data set With PCA algorithm results test_pca
scaler.fit(Testset)
scaled_test = scaler.transform(Testset)

pca = PCA(n_components=1)
pca.fit(scaled_test)

test_pca = pca.transform(scaled_test)

#Learning Part
n_iter = 40
eta0 = 0.1
random_state = 0
ppn = Perceptron(eta0=eta0, random_state=random_state)
ppn.fit(x_pca, sonuc)
#prediction with Preceptron algorithm
y_pred = ppn.predict(test_pca)

#printing the y_pred to Submission.csv
with open('..\Data\Submission.csv', 'w', newline='') as submission:
    sub_writer = csv.writer(submission)
    sub_writer.writerow(['ID', 'Predicted'])
    for i in range(80):
        if y_pred[i] == 1:
            sub_writer.writerow([i+1, '1'])
        else:
            sub_writer.writerow([i+1, '0'])

