# -*- coding: utf-8 -*-

import sklearn.datasets as DS_
import sklearn.linear_model as sklm
import sklearn.metrics as metrics
import sklearn.model_selection as skml

# load data ---------------------------------------------------------------
iris = DS_.load_iris()
X, y = iris.data, iris.target


# split data --------------------------------------------------------------
xtrain, xtest, ytrain, ytest = skml.train_test_split(X, y, test_size=0.2)

# preprocessing -----------------------------------------------------------

# model -------------------------------------------------------------------
model = sklm.LinearRegression()
model.fit(xtrain, ytrain)

# evaluation --------------------------------------------------------------
ypred = model.predict(xtest)
r2_ = metrics.r2_score(ytest, ypred)
print('r2_', r2_)




