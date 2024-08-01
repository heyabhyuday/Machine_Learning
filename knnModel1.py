import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler

import sklearn.datasets

trainset = pd.read_csv('Train_new.csv')
testset = pd.read_csv('Test_new.csv')

acc = []
t_acc = []

kf = KFold(n_splits=5, shuffle=True, random_state=1)

for train_index_vals, test_index_vals in kf.split(trainset):
    model = KNeighborsClassifier()

    x_train = trainset.iloc[train_index_vals].values
    y_train = trainset.iloc[train_index_vals].values
    x_test = trainset.iloc[test_index_vals].values
    y_test = trainset.iloc[test_index_vals].values

    scaler = MinMaxScaler()
    # Fitting on the training data
    scaler.fit(x_train)
    # transforming training data
    train_x = scaler.transform(x_train)
    # transforming test data
    test_x = scaler.transform(x_test)

    # train
    model.fit(x_train, y_train)
    # make prediction
    y_pred = model.predict(x_test)

    # evaluate the accuracy
    acc.append(accuracy_score(y_test, y_pred))
    # accuracy over training data
    t_acc.append(accuracy_score(y_train, model.predict(x_train)))

print(acc)
cv_acc = np.mean(acc)
print(cv_acc)
