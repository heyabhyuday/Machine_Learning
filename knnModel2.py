import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.inspection import DecisionBoundaryDisplay

import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler

import sklearn.datasets

#Making pandas dataframes from the files. Seperating data and labels.
trainset = pd.read_csv('Train_new.csv')
traindata = trainset.iloc[:, :-1]
trainlabels = trainset.iloc[:, -1]

testset = pd.read_csv('Test_new.csv')
testdata = testset.iloc[:, :-1]
testlabels = testset.iloc[:, -1]

acc = []
t_acc = []
k_acc = {}

for k in range(1, 20, 2):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index_vals, test_index_vals in kf.split(trainset):
        # Initializing model
        model = KNeighborsClassifier(n_neighbors=k)

        # Splitting the Train csv into training and test sets
        x_train = traindata.iloc[train_index_vals].values
        y_train = trainlabels.iloc[train_index_vals].values
        x_test = traindata.iloc[test_index_vals].values
        y_test = trainlabels.iloc[test_index_vals].values

        # Scaling and transforming data
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        train_x = scaler.transform(x_train)
        test_x = scaler.transform(x_test)

        # Training the model
        model.fit(x_train, y_train)
        # Making predictions
        y_pred = model.predict(x_test)

        # Evaluating accuracy
        acc.append(accuracy_score(y_test, y_pred))
        t_acc.append(accuracy_score(y_train, model.predict(x_train)))

    # print(f"\nAccuracies after 5-fold CV for k = {k}:")
    # for i in range(5):
    #     print(f"{(2 * i) + 1}.", acc[i])

    cv_acc = np.mean(acc)
    # print(f"\nMean accuracy of 5-fold CV for k = {k}:\n", cv_acc)
    k_acc[k] = cv_acc

print("\nMean accuracies of 5-fold CV for different k values:")
for f in k_acc:
    if f < 11:
        print(f, " : ", k_acc[f])
    else:
        print(f, ": ", k_acc[f])

max_k = max(zip(k_acc.values(), k_acc.keys()))[1]
print("The best accuracy is seen at k =", max_k)
print("\nRunning the model with k =", max_k, "...")
#Setting model with best k
model = KNeighborsClassifier(n_neighbors=max_k)

#Getting training and test sets from the csv files
x_train = traindata.values
y_train = trainlabels.values
x_test = testdata.values
y_test = testlabels.values

# Scaling and transforming data
scaler = MinMaxScaler()
scaler.fit(x_train)
train_x = scaler.transform(x_train)
test_x = scaler.transform(x_test)

# Training the model
model.fit(x_train, y_train)
# Making predictions
y_pred = model.predict(x_test)

# Evaluating accuracy
final_acc = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, model.predict(x_train))

print(f"\nAccuracy of the model with k = {max_k} is", final_acc)

#Evaluating F1 score
y_true = testlabels
f1 = f1_score(y_true, y_pred, zero_division=1)
print("F1 score for this dataset is", f1)
