#!/usr/bin/env python

import pandas as pd
import xgboost as xgb
import json
import shutil
import time
from sklearn.metrics import roc_auc_score
import os
from torch.utils.tensorboard import SummaryWriter

# Specify training params
data_path = '/media/ziyuexu/Data/HIGGS/HIGGS_UCI.csv'
site_num = 5
tree_num = 100
round_num = int(tree_num/site_num)

# Set record paths
model_path_root = 'Model_Bagging/'
model_path = model_path_root + str(site_num) + '_Sites_' + str(tree_num) + '_Trees/'
os.makedirs(model_path)
# Set mode file paths
model_path_full = model_path + 'bagging.json'
model_path_sub_pre = model_path + 'bagging_site_temp_'
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
X_higgs_valid = X_higgs.iloc[0:valid_num, :]
y_higgs_valid = y_higgs.iloc[0:valid_num]
dmat_valid = xgb.DMatrix(X_higgs_valid, label=y_higgs_valid)
dmat_train = []
model_path_sub = []
# split to multi_site data
site_size = int((total_data_num - valid_num) / site_num)
for site in range(site_num):
    idx_start = valid_num + site_size * site
    idx_end = valid_num + site_size * (site + 1)
    X_higgs_train = X_higgs.iloc[idx_start:idx_end, :]
    y_higgs_train = y_higgs.iloc[idx_start:idx_end]
    dmat_train.append(xgb.DMatrix(X_higgs_train, label=y_higgs_train))
    model_path_sub.append(model_path_sub_pre + str(site) + '.json')

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

param_bagging = {}
param_bagging['objective'] = param['objective']
param_bagging['eta'] = 0.1
param_bagging['max_depth'] = param['max_depth']
param_bagging['eval_metric'] = param['eval_metric']
param_bagging['nthread'] = param['nthread']
param_bagging['num_parallel_tree'] = site_num

# xgboost training
start = time.time()
for round in range(round_num):
    # Train individual sites' models
    # Starting from "global model" with bagging boosting
    # Saving to individual files
    if os.path.exists(model_path_full):
        # Load global model under bagging param setting
        bst_bagging = xgb.Booster(param_bagging, model_file=model_path_full)
        # Validate global model first
        y_pred = bst_bagging.predict(dmat_valid)
        roc_bagging = roc_auc_score(y_higgs[0:1000000], y_pred)
        print(f"Round: {bst_bagging.num_boosted_rounds()} global bagging model testing AUC {roc_bagging}")
        writer.add_scalar('AUC', roc_bagging, round-1)
    # For each site, load global model to set base margin if global model exist
    for site in range(site_num):
        print(f"Round: {round} Site {site} ", end='')
        if os.path.exists(model_path_full):
            # Compute margin on site's data
            ptrain = bst_bagging.predict(dmat_train[site], output_margin=True)
            pvalid = bst_bagging.predict(dmat_valid, output_margin=True)
            # Set margin
            dmat_train[site].set_base_margin(ptrain)
            dmat_valid.set_base_margin(pvalid)
        # Boost a tree under tree param setting
        bst = xgb.train(param, dmat_train[site], num_boost_round=1,
                        evals=[(dmat_valid, 'validate'), (dmat_train[site], 'train')])
        bst.save_model(model_path_sub[site])

    if not os.path.exists(model_path_full):
        # Initial, copy from tree 1
        shutil.copy(model_path_sub[0], model_path_full)
        # Remove the first tree
        with open(model_path_full) as f:
            json_bagging = json.load(f)
        json_bagging['learner']['gradient_booster']['model']['trees'] = []
        with open(model_path_full, 'w') as f:
            json.dump(json_bagging, f, separators=(',', ':'))

    with open(model_path_full) as f:
        json_bagging = json.load(f)
    # Append this round's trees to global model tree list
    for site in range(site_num):
        with open(model_path_sub[site]) as f:
            json_single = json.load(f)
        # Always 1 tree, so [0]
        append_info = json_single['learner']['gradient_booster']['model']['trees'][0]
        append_info['id'] = round * site_num + site
        json_bagging['learner']['gradient_booster']['model']['trees'].append(append_info)
        json_bagging['learner']['gradient_booster']['model']['tree_info'].append(0)
    json_bagging['learner']['attributes']['best_iteration'] = str(round)
    json_bagging['learner']['attributes']['best_ntree_limit'] = str(site_num*(round+1))
    json_bagging['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'] = str(site_num*(round+1))
    # Save the global bagging model
    with open(model_path_full, 'w') as f:
        json.dump(json_bagging, f, separators=(',', ':'))

end = time.time()
lapse_time = end - start
print(f"Training time: {lapse_time}")

# test model
bst_bagging = xgb.Booster(param_bagging, model_file=model_path_full)
y_pred = bst_bagging.predict(dmat_valid)
roc_bagging = roc_auc_score(y_higgs[0:1000000], y_pred)
print(f"Bagging model: {roc_bagging}")
writer.add_scalar('AUC', roc_bagging, round_num-1)

writer.close()