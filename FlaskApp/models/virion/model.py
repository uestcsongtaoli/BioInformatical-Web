# coding: utf-8

import itertools as iters
from time import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.feature_selection import f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pylab as plt
from math import log2 
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
import os
absolute_path = os.path.abspath('.')

def entropy(data):
    length,dataDict=len(data),{} 
    for b in data:  
        try:dataDict[b]+=1  
        except:dataDict[b]=1  
    entropy=sum([-d/length*log2(d/length) for d in list(dataDict.values())])
    return entropy

def informationgain(data,label):
    informationgain = []
    la = entropy(label)
    print(la)
    for j in range(data.shape[1] ):
        feature = data[:,j]
        for a in set(feature):
            ent = []
            op = []
            ne = []
            for k in range(len(feature)):
                if feature[k] >= a:
                    op.append(label[k])
                else:
                    ne.append(label[k])
            if len(op) == 0 or len(ne) == 0:
                ent.append(la)
            else:
                ent.append(len(op)*entropy(op)/len(label) + len(ne)*entropy(ne)/len(label))
        informationgain.append(la-min(ent))
    return informationgain

##计算三肽特征
def statisPsi_3(seqs,protein,gap1,gap2):
    psi = np.zeros(len(seqs))
    loops = len(protein) - gap1 - gap2 - 2
    for start in range(loops):
        dipeptide = protein[start] + protein[start + gap1 + 1] + protein[start + 2 + gap1 + gap2]
        index = seqs.index(dipeptide)
        psi[index] += 1
    psi = np.array(psi)
    psi = psi / sum(psi)
    return psi

# get gap dipeptide features psi matrix",
def all_psi(dataset,gap1,gap2):
    SAA = ('ACDEFGHIKLMNPQRSTVWY')
    DIPEPTIDE = []

    for dipeptide in iters.product(SAA, repeat=3):
        DIPEPTIDE.append(''.join(dipeptide))
    gap_psi = np.zeros((len(dataset), len(DIPEPTIDE)))
    for idx in range(len(dataset)):
        gap_psi[idx] = statisPsi_3(DIPEPTIDE, dataset[idx], gap1,gap2)
    return gap_psi




def predict(txtpath):
    result = []
    with open(absolute_path + '/FlaskApp/models/virion/virion.txt') as file:
        train_tdata = [line.strip() for line in file if '>' != line[0]]
    with open(absolute_path + '/FlaskApp/models/virion/non-virion.txt') as file:
        train_fdata = [line.strip() for line in file if '>' != line[0]]
    with open(txtpath,"r") as file:
        predict_data = [line.strip() for line in file if '>' != line[0]]

    SAA = ('ACDEFGHIKLMNPQRSTVWY')
    DIPEPTIDE = []

    for dipeptide in iters.product(SAA, repeat=3):
        DIPEPTIDE.append(''.join(dipeptide))

    label = pd.Series([1 for i in range(len(train_tdata))]+ [0 for i in range(len(train_fdata))])
    label = label.as_matrix()
    gap1 = 2
    gap2 = 1
    gap_T = all_psi(train_tdata,gap1,gap2)
    gap_F = all_psi(train_fdata,gap1,gap2)
    predict_data = all_psi(predict_data,gap1,gap2)
    
    dataAll = np.row_stack((gap_T, gap_F))
    f = informationgain(dataAll,label)
    ##因为三肽特征较为稀疏，会有一些全为nan的列，要对这些列进行处理
    a = np.array(f)
    nan_count = np.sum(a != a)
    print(nan_count)
    ##将nan替换为0
    f1 = np.nan_to_num(a)
    ##将F值从大到小排序，获得相应的位置序号
    f_order = np.argsort(f1).tolist()[::-1]
    data = dataAll[:,f_order[0:771]]
    predict_data = predict_data[:,f_order[0:771]]
    
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    predict_data = scaler.transform(predict_data)
    
    C_range = np.logspace(15, 5, 11, base=2)
    gamma_range = np.logspace(-15, -25, 11, base=2)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=5)  # 基于交叉验证的网格搜索
    
    grid = grid.fit(data,label)
    clf = SVC(C = grid.best_params_['C'],gamma=grid.best_params_['gamma'])
    clf.fit(data,label)
    result = clf.predict(predict_data).tolist()
    return result

if(__name__=='__main__'):

    txtpath = './1/virion.txt'
    result = predict('./non-virion.txt')
    print(result)
