# Please make sure scikit-learn is included the conda_dependencies.yml file.
from comet_ml import Experiment
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

# initialize comet_ml
experiment = Experiment(api_key="3or0Re1ghS53CqBBU0d8apkwRkET4dEwlrYINrZCFynUHJEycmBou6pPv9yRtl1E", log_code=True)

print('Python version: {}'.format(sys.version))

# load Iris dataset from a DataPrep package as a pandas DataFrame
iris = pd.read_csv("iris_raw.csv")
print('Iris dataset shape: {}'.format(iris.shape))

# load features and labels
X, Y = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values, iris['species'].values

# add n more random features to make the problem harder to solve
# number of new random features to add
n = 40
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, n)]

# split data 65%-35% into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)

# change regularization rate and you will likely get a different accuracy.
reg = 0.4
# load regularization rate from argument if present
if len(sys.argv) > 1:
    reg = float(sys.argv[1])
# experiment.log_parameter("Regularization",reg)
print("Regularization rate is {}".format(reg))

# train a logistic regression model on the training set
clf1 = LogisticRegression(C=1 / reg)
clf1 = clf1.fit(X_train, Y_train)
print(clf1)

# evaluate the test set
accuracy = clf1.score(X_test, Y_test)
print("Accuracy is {}".format(accuracy))

# calculate and log precesion, recall, and thresholds, which are list of numerical values
y_scores = clf1.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(Y_test, y_scores[:, 1], pos_label='versicolor')

prec = precision[len(precision)//2]
reca= recall[len(recall)//2]
thresh = thresholds[len(thresholds)//2]
# experiment.log_metric("Precision", prec)
# experiment.log_metric("Recall", reca)
# experiment.log_metric("Thresholds", thresh)

print('Precision at 0.5: {}'.format(prec))
print('Recall at 0.5: {}'.format(reca))
print('Threshold at 0.5: {}'.format(thresh))

# predict on a new sample
X_new = [[3.0, 3.6, 1.3, 0.25]]
print('New sample: {}'.format(X_new))

# add random features to match the training data
X_new_with_random_features = np.c_[X_new, random_state.randn(1, n)]

# score on the new sample
pred = clf1.predict(X_new_with_random_features)
print('Predicted class is {}'.format(pred))
