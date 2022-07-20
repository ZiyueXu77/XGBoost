#!/usr/bin/env python

import pandas as pd
import xgboost as xgb
import time
from sklearn.metrics import roc_auc_score
import os
from torch.utils.tensorboard import SummaryWriter

# Specify training params
data_path = '/media/ziyuexu/Data/HIGGS/HIGGS_UCI.csv'
site_num = 5
tree_num = 5
round_num = int(tree_num/site_num)

# Set record paths
model_path_root = 'Model_RF/'
model_path = model_path_root + str(site_num) + '_Sites_' + str(tree_num) + '_Trees/'
os.makedirs(model_path)
# Set mode file paths
model_path_ref = model_path + 'random_forest.json'
# Set tensorboard output
writer = SummaryWriter(model_path)

# Load data
start = time.time()
higgs = pd.read_csv(data_path, header=None)
print(higgs.info())
print(higgs.head())
total_data_num = higgs.shape[0]
valid_num = 1000000
print(f"Total data count: {total_data_num}")
# split to feature and label
X_higgs = higgs.iloc[:, 1:]
y_higgs = higgs.iloc[:, 0]
print(y_higgs.value_counts())
end = time.time()
lapse_time = end - start
print(f"Data loading time: {lapse_time}")

# construct xgboost DMatrix
# split to validation and multi-site training
dmat_higgs = xgb.DMatrix(X_higgs, label=y_higgs)
dmat_valid = dmat_higgs.slice(X_higgs.index[0:valid_num])
dmat_train = dmat_higgs.slice(X_higgs.index[valid_num:])

# setup parameters for xgboost with RF
# use logistic regression loss for binary classification
# learning rate 0.1 max_depth 5
# use auc as metric
param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 8
param['eval_metric'] = 'auc'
param['nthread'] = 16
param['num_parallel_tree'] = site_num
param['subsample'] = 1 / site_num

# xgboost training
start = time.time()
for round in range(round_num):
    # Train a random forest model
    if os.path.exists(model_path_ref):
        # Validate the last round's model
        bst_RF_last = xgb.Booster(param, model_file=model_path_ref)
        y_pred = bst_RF_last.predict(dmat_valid)
        roc_RF = roc_auc_score(y_higgs[0:1000000], y_pred)
        print(f"Round: {bst_RF_last.num_boosted_rounds()-1} random forest model testing AUC {roc_RF}")
        writer.add_scalar('AUC', roc_RF, round-1)
        # Train new model
        print(f"Round: {round} Random Forest ", end='')
        bst_RF = xgb.train(param, dmat_train, num_boost_round=1, xgb_model=model_path_ref,
                           evals=[(dmat_valid, 'validate'), (dmat_train, 'train')])
    else:
        # Round 0
        print(f"Round: {round} Random Forest ", end='')
        bst_RF = xgb.train(param, dmat_train, num_boost_round=1,
                           evals=[(dmat_valid, 'validate'), (dmat_train, 'train')])
    bst_RF.save_model(model_path_ref)

end = time.time()
lapse_time = end - start
print(f"Training time: {lapse_time}")

# test final model
bst_RF = xgb.Booster(param, model_file=model_path_ref)
y_pred = bst_RF.predict(dmat_valid)
roc_RF = roc_auc_score(y_higgs[0:1000000], y_pred)
print(f"Random Forest model: {roc_RF}")
writer.add_scalar('AUC', roc_RF, round_num-1)

writer.close()