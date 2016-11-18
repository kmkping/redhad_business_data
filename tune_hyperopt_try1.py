import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedKFold,KFold,train_test_split
from scipy.stats import randint, uniform
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import LabelKFold
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

import datetime
import random
from operator import itemgetter
import time
import copy

from scipy.io import mmread

np.random.seed(333)

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import os
import sys
sys.stdout = open('tune_hyperopt_try1.txt', 'w', 1)

def cvtest(i,params,xgboost_prob,num_round):
    
    plst = list(params.items())
    num_boost_round = num_round
    # pass the indexes to your training and validation data

    xgtrain = dtrain.slice(cv[i][0])
    xgval = dtrain.slice(cv[i][1])

    # define a watch list to observe the change in error f your training and holdout data

    evals = [(xgtrain, 'train'), (xgval, 'eval')]

    model = xgb.train(params, xgtrain, num_boost_round, early_stopping_rounds=30, evals=evals, verbose_eval=10)

    pred_train = model.predict(xgval, ntree_limit = model.best_ntree_limit)
    xgboost_prob[cv[i][1]] = pred_train
    res = roc_auc_score(xgval.get_label(), pred_train)
    print(res)
    return (res, model.best_iteration)

def score(params):
    print("Training with params : ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    results_auc = np.repeat(0.0, k)
    trees = np.repeat(0.0, k)
    xgboost_prob = np.zeros(dtrain.num_row(), dtype=np.float64)
    for i in range(k):
        (results_auc[i], trees[i]) = cvtest(i,params,xgboost_prob,num_round)

    score_auc = roc_auc_score(dtrain.get_label(), xgboost_prob)
    score = 1 - score_auc
    print("\tAuc_kfold: {0}".format(score_auc))
    print("\tAuc_avg: {0}".format(sum(results_auc)/len(results_auc)))
    print("\tTrees mean: {0}".format(sum(trees)/len(trees)))
    print("\tTrees_kfold: {0}\n\n".format(trees))
    return {'loss': score, 'status': STATUS_OK}

def optimize(random_state=5):
    space = {
             'n_estimators' : 100000,
             'eta' : 0.3,
             'max_depth' : hp.quniform('max_depth', 2, 25, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 12, 1),
             'subsample' : hp.uniform('subsample', 0, 1),
             'colsample_bytree' : hp.uniform('colsample_bytree', 0, 1),
             'colsample_bylevel' : hp.uniform('colsample_bylevel', 0, 1),
             'gamma' : hp.uniform('gamma', 0, 1),
             'lambda': hp.uniform('lambda', 0, 5),
             'alpha': hp.uniform('alpha', 0, 5),
             'eval_metric': 'auc',
             'objective': 'binary:logistic',
             'nthread' : 20,
             'silent' : 1,
             'seed' : random_state
             }

    best = fmin(score, space, algo=tpe.suggest, max_evals=500)

    print(best)

dtrain = xgb.DMatrix('svmlight_try2/dtrain.data')
dtest = xgb.DMatrix('svmlight_try2/dtest.data')

act_train_data = pd.read_csv("redhat_data_new/act_train_new_try2.csv",dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])

k = 5
cv = LabelKFold(act_train_data['people_id'], n_folds=k)
cv = list(cv)

#Trials object where the history of search will be stored
trials = Trials()

optimize()