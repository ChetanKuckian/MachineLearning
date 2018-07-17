# -*- coding: utf-8 -*-
"""
@author: F34R
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


data = load_breast_cancer()

labels_name = data['target_names']
labels = data['target']
features_name = data['feature_names']
features = data['data']
#print(label_names)
#print(labels[0])
#print(feature_names[0])
#print(features[0])

train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=50)


lr = LogisticRegression()
model = lr.fit(train, train_labels)


preds = model.predict(test)
print(preds)
print(accuracy_score(test_labels, preds))
