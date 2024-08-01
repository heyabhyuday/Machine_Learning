import os
import numpy as np
from skimage import io, color, feature, util
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

st = time.time()

X = []  # Features
y = []  # Labels

# Distance and angle offsets for GLCM
distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Iterate through every image found in the specified directory
directory = r'C:\Users\Abhyuday\FLAME\Sem 4\CSIT334\Steel Surfaces\NEU-DET\train\images'
for file in os.listdir(directory):
    if file.endswith('.jpg'):
        # Convert image to grayscale
        image = io.imread(os.path.join(directory, file))
        gs = util.img_as_ubyte(color.rgb2gray(image))

        # Calculate Gray Level Co-occurrence Matrix (GLCM)
        glcm = feature.graycomatrix(gs, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # Calculate GLCM properties
        contrast = feature.graycoprops(glcm, 'contrast')[0][0]
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0][0]
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0][0]
        energy = feature.graycoprops(glcm, 'energy')[0][0]
        correlation = feature.graycoprops(glcm, 'correlation')[0][0]

        # Add features and labels
        X.append([contrast, dissimilarity, homogeneity, energy, correlation])
        y.append(file.split('_')[0])  # In line with dataset's image naming convention

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=42)

Src = StandardScaler()
dTree = DecisionTreeClassifier()

pipe = Pipeline([
    ('scaler', Src),
    ('DecisionTree', dTree)
])

param_grid = [{
    'regressor__max_depth': [2, 3, 4, 5, 6],
    'regress__min_samples_split': [2, 3, 5, 10, 15]
}]

grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5)

DTree = grid.fit(X, y)

plot_tree(DTree.best_estimator_['regressor'], max_depth=5,
          impurity=True,
          feature_names='X',
          precision=1,
          filled=True)

print('Best hyperparameters:', grid.best_params_)

accuracy = grid.score(X_test, y_test)
print('Accuracy:', accuracy)

for i, test_img in enumerate(X_test):
    predicted_label = grid.predict([test_img])[0]
    file = os.path.basename(y_test[i])
    print(f'For file {file}, predicted label is {predicted_label}')
