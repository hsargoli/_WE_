
# -*- coding: utf-8 -*-

import sklearn.datasets as DS_
import sklearn.linear_model as sklm
import sklearn.metrics as metrics
import sklearn.model_selection as skml
# import warnings
# warnings.filterwarnings('ignore')
import sklearn.preprocessing as Prep
import sklearn.ensemble as ensemble
import numpy as np

# load data ---------------------------------------------------------------
breast_cancer = DS_.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target


# split data --------------------------------------------------------------
xtrain, xtest, ytrain, ytest = skml.train_test_split(X, y, test_size=0.2)

# preprocessing -----------------------------------------------------------
scaler = Prep.MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# model -------------------------------------------------------------------
model = sklm.LogisticRegression(max_iter=10_000,
                                C=100,
                                solver='liblinear')


cv_ = skml.cross_validate(sklm.LogisticRegression(max_iter=10_000,
                                C=100,
                                solver='liblinear'),
                          xtrain,ytrain, 
                          cv=10)

print('>>> cv score avg: ',cv_['test_score'].mean())


model.fit(xtrain, ytrain)


# evaluation --------------------------------------------------------------
ypred = model.predict(xtest)
f1_score = metrics.f1_score(ytest, ypred)
print('f1_score', f1_score)

    