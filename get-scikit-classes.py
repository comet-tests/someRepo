from sklearn.utils.testing import all_estimators
from sklearn import base
estimators = all_estimators()

for name, class_ in estimators:
    if issubclass(class_, base.ClassifierMixin):
        print(class_)

for name, class_ in estimators:
    if issubclass(class_, base.RegressorMixin):
        print(class_)

exit(0);
