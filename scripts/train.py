# Script for training machine learning model
# Author: Md Mahmudulla Hassan
# Last modified: 03/25/2019

import pickle
import numpy as np
import pandas as pd
import glob
import os
from tqdm import *
import random
import json

DATA_DIR = '../data'
MODEL_DIR = '../models'
if not os.path.isdir(MODEL_DIR): os.makedirs(MODEL_DIR)

print("Reading gene dictionary")
with open(os.path.join(DATA_DIR, 'gene_dict.json'), 'r') as f:
    gene_dict = json.load(f)

print("Reading tpatf features")
data_files = glob.glob(os.path.join(DATA_DIR, 'npy_data_tpatf/*.npy'))
data_genes = [os.path.split(i)[1][:-4] for i in data_files]

# Filter out the genes that we don't have a sequence for
data_genes = [i for i in data_genes if i in gene_dict.keys()]
print("Number of genes: " + str(len(data_genes)))
print("Extracting features...")
# First sample
indices = {} # records the indices of the gene samples
data = np.load(os.path.join(DATA_DIR, 'npy_data_tpatf/' + data_genes[0] + '.npy'))
indices[data_genes[0]] = [i for i in range(data.shape[0])]
protein_feature = gene_dict[data_genes[0]]['features']
features = np.hstack((np.tile(protein_feature, (data.shape[0], 1)), data)).astype(np.float32)

for i in tqdm(range(1, len(data_genes))):
    _data = np.load(os.path.join(DATA_DIR, 'npy_data_tpatf/' + data_genes[i] + '.npy'))
    indices[data_genes[i]] = [i for i in range(features.shape[0], features.shape[0] + _data.shape[0])]
    _protein_feature = gene_dict[data_genes[i]]['features']
    _features = np.hstack((np.tile(_protein_feature, (_data.shape[0], 1)), _data)).astype(np.float32)
    features = np.vstack((features, _features))

data_x = features[:, :-1]
data_y = features[:, -1:]

def random_scaffold(data_x, data_y, indices_dict, test_size=0.2):
    test_sample_count = round(data_x.shape[0] * test_size)
    test_sample_indices = []
    indices_list = list(indices_dict.values())
    taken = [] # Keeps track of the selected protein index

    while len(test_sample_indices) < test_sample_count:
        _index_of_protein = random.randrange(0, len(indices_list))
        if _index_of_protein not in taken:
            taken.append(_index_of_protein)
            test_sample_indices.extend(indices_list[_index_of_protein])
    
    train_sample_indices = list(set([i for i in range(data_x.shape[0])]) - set(test_sample_indices))

    return data_x[train_sample_indices], data_y[train_sample_indices], data_x[test_sample_indices], data_y[test_sample_indices]


from evaluation_metrics import rmse, pearson, spearman, f1, ci, average_AUC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib


rmse_list = []
pearson_list = []
spearman_list = []
f1_list = []
ci_list = []
auc_list = []

exp_count = 20
print("Training random forest models by leave-on-out method")
print("Total number of experiments/models: {}".format(exp_count))
for i in range(exp_count):
    print("EXPERIMENT {}: ".format(i+1), end=' ')
    
    train_x, train_y, test_x, test_y = random_scaffold(data_x, data_y.ravel(), indices, test_size=0.1)
    model = RandomForestRegressor(n_estimators=1000, 
                              max_features=5, 
                              min_samples_leaf=1, 
                              max_depth=50, 
                              n_jobs=-1, 
                              random_state=1, 
                              verbose=False)
    model.fit(train_x, train_y)
    joblib.dump(model, os.path.join(MODEL_DIR, 'model_' + str(i+1)))
    
    #train_predict = model.predict(train_x)
    #train_r2 = r2_score(y_pred=train_predict, y_true=train_y)
    test_predict = model.predict(test_x)
    #test_r2 = r2_score(y_pred=test_predict, y_true=test_y)
    #print("  TRAIN R2: {:.2}, TEST R2: {:.2}".format(train_r2, test_r2))
    
    RMSE = rmse(test_y, test_predict)
    rmse_list.append(RMSE)
    PEARSON = pearson(test_y, test_predict)
    pearson_list.append(PEARSON)
    SPEARMAN = spearman(test_y, test_predict)
    spearman_list.append(SPEARMAN)
    F1 = f1(test_y, test_predict)
    f1_list.append(F1)
    CI = ci(test_y, test_predict)
    ci_list.append(CI)
    AVG_AUC = average_AUC(test_y, test_predict)
    auc_list.append(AVG_AUC)

    print("RMSE: {:.2f} PEARSON: {:.2f} SPEARMAN: {:.2f}, F1: {:.2f}, CI: {:.2f}, AVG AUC: {:.2f}".format(RMSE,
                                                                                                          PEARSON,
                                                                                                          SPEARMAN, 
                                                                                                          F1, 
                                                                                                          CI, 
                                                                                                          AVG_AUC))
    
print("MEAN RMSE: {:.2}, PEARSON: {:.2}, SPEARMAN: {:.2}, F1: {:.2}, CI: {:.2}, AVG AUC: {:.2}".format(sum(rmse_list)/len(rmse_list),
                                                                                                       sum(pearson_list)/len(pearson_list),
                                                                                                       sum(spearman_list)/len(spearman_list),
                                                                                                       sum(f1_list)/len(f1_list),
                                                                                                       sum(ci_list)/len(ci_list),
                                                                                                       sum(auc_list)/len(auc_list)))


# ROUND 1 BEST <br>
# rmse 1.0421, pearson 0.544, spearman 0.5523, f1 0.5875, ci 0.711, average AUC 0.7975
