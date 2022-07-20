#!/usr/bin/env python

import pandas as pd
import xgboost as xgb
import time
from sklearn.metrics import roc_auc_score
import os
from torch.utils.tensorboard import SummaryWriter

# Specify training params
data_path = '/media/ziyuexu/Data/HIGGS/HIGGS_UCI.csv'
site_num = 20
tree_num = 100
round_num = int(tree_num/site_num)

# Set mode file paths
model_path_root = 'Model_Cyclic/'
model_path_root = model_path_root + str(site_num) + '_Sites_' + str(tree_num) + '_Trees/'
os.makedirs(model_path_root)
# Set mode file paths
model_path = model_path_root + 'cyclic.json'
# Set tensorboard output
writer = SummaryWriter(model_path_root)

# Load data
start = time.time()
higgs = pd.read_csv(data_path, header=None)
print(higgs.info())
print(higgs.head())
total_data_num = higgs.shape[0]
valid_num = 1000000
print(f"Total data count: {total_data_num}")
# split to feature and label
X_higgs = higgs.iloc[:,1:]
y_higgs = higgs.iloc[:,0]
print(y_higgs.value_counts())
end = time.time()
lapse_time = end - start
print(f"Data loading time: {lapse_time}")

# construct xgboost DMatrix
# split to validation and multi-site training
dmat_higgs = xgb.DMatrix(X_higgs, label=y_higgs)
dmat_valid = dmat_higgs.slice(X_higgs.index[0:valid_num])
dmat_train = []
# split to multi_site data
site_size = int((total_data_num - valid_num) / site_num)
for site in range(site_num):
    idx_start = valid_num + site_size * site
    idx_end = valid_num + site_size * (site + 1)
    dmat_train.append(dmat_higgs.slice(X_higgs.index[idx_start:idx_end]))

# setup parameters for xgboost
# use logistic regression loss for binary classification
# learning rate 0.1 max_depth 5
# use auc as metric
param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 8
param['eval_metric'] = 'auc'
param['nthread'] = 16

# xgboost training
start = time.time()
for round in range(round_num):
    # Train a cyclic model
    for site in range(site_num):
        if os.path.exists(model_path):
            bst_last = xgb.Booster(param, model_file=model_path)
            y_pred = bst_last.predict(dmat_valid)
            roc = roc_auc_score(y_higgs[0:1000000], y_pred)
            print(f"Trees {round*site_num + site - 1} model testing AUC {roc}")
            writer.add_scalar('AUC', roc, round*site_num + site - 1)
            # Train new model
            print(f"Round: {round} Site: {site} ", end='')
            bst = xgb.train(param, dmat_train[site], num_boost_round=1, xgb_model=model_path,
                            evals=[(dmat_valid, 'validate'), (dmat_train[site], 'train')])
        else:
            # Round 0
            print(f"Round: {round} Site: {site} ", end='')
            bst = xgb.train(param, dmat_train[site], num_boost_round=1,
                            evals=[(dmat_valid, 'validate'), (dmat_train[site], 'train')])
        bst.save_model(model_path)

end = time.time()
lapse_time = end - start
print(f"Training time: {lapse_time}")

# test model
bst = xgb.Booster(param, model_file=model_path)
y_pred = bst.predict(dmat_valid)
roc = roc_auc_score(y_higgs[0:1000000],y_pred)
print(f"Cyclic model: {roc}")
writer.add_scalar('AUC', roc, tree_num-1)

writer.close()
