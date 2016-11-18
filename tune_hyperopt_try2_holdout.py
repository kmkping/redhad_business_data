from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn import preprocessing

import numpy as np
import pandas as pd
np.random.seed(22)

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import os
import sys
sys.stdout = open('tune_hyperopt_try2_holdout.txt', 'w', 1)
import xgboost as xgb

def score(params):
    print("Training with params : ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    watchlist = [(dtrain, 'train'),(dval, 'eval')]
    model = xgb.train(params, dtrain, num_round, early_stopping_rounds=50, evals=watchlist, verbose_eval=0)
    #pred_val = model.predict(dval, ntree_limit=model.best_ntree_limit)
    #score_val = roc_auc_score(labels, pred_val)
    score_val = model.best_score
    score = 1 - score_val
    print("\tAuc_val: {0}\n\n".format(score_val))
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

dtrain = xgb.DMatrix('to_r_n_back/dtrain.data')
dval = xgb.DMatrix('to_r_n_back/dtest.data')
yval = (pd.read_csv('to_r_n_back/val1_target.csv')).outcome.values
labels = yval
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)
dval.set_label(labels)

#Trials object where the history of search will be stored
trials = Trials()

optimize()