import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import math

iris = datasets.load_iris()
X = iris.data()
y = iris.target()

print(y)
