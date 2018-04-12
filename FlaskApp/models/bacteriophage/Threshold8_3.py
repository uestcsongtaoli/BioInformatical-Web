# -*- coding:utf-8 -*-
import matplotlib
import numpy
import pandas
from matplotlib import pyplot
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

positive = 99

label = [1 if i < positive else 0 for i in range (307)]


def load_data(path):

    dataframe = pandas.read_csv (path, header=0, index_col=None)
    df = dataframe.replace(0, numpy.nan)
    df.dropna (axis=1, how='all')
    df = df.replace(numpy.nan, 0)
    data = df.values

    return data


def find_best_split(Data, label):

    index = numpy.argwhere(Data > 0)
    data = Data[index]
    median = numpy.median(data)

    Data = numpy.piecewise(Data, [(Data > median) & (Data != 0),(Data <= median) & (Data != 0), Data == 0], [2, 1, 0])

    return median, Data


def metric(data, label):

    return metrics.f1_score(label, data, pos_label=1) + metrics.f1_score(label, data, pos_label=0)


def update_threshold(data, threshold, idx_score):

    for i in range(len(threshold)):

        data[:, i] = numpy.piecewise(data[:, i], [data[:, i] == 0, (data[:, i] <= threshold[i]) & (data[:, i] != 0), (data[:, i] > threshold[i]) & (data[:, i] != 0)], [0, 1, 2])

    new = numpy.copy(data)

    for j, idx in enumerate(idx_score):
        data[:, j] = new[:, idx]

    return data


def sort(data, label):

    new_data = numpy.copy(data)
    score = numpy.zeros(data.shape[1])

    for i in range(data.shape[1]):

        data0 = numpy.copy(data[:, i])
        data1 = numpy.copy(data[:, i])

        data0[data0 == 2] = 0
        data1[data1 == 2] = 1

        if metric(data0, label) >= metric(data1, label):
            score[i] = metric(data0, label)
        else:
            score[i] = metric(data1, label)

    idx_score = numpy.argsort(-score)
    for j, idx in enumerate(idx_score):
        new_data[:, j] = data[:, idx]

    return new_data, idx_score


def Threshold(data, label):

    threshold = numpy.zeros(data.shape[1])

    flag = 0
    for i in range(data.shape[1]):

        threshold[i], new_data = find_best_split(data[:, i], label)
        new_data = numpy.reshape(new_data, (len(new_data), 1))

        if flag == 1:
            new = numpy.concatenate((new, new_data), axis=1)
        else:
            new = new_data
            flag = 1

    new, idx_score = sort(new, label)

    # df = pandas.DataFrame (new)
    # df.to_csv (r'after_threshold.csv', header=None, index=None)

    return new, threshold, idx_score


if __name__ == '__main__':
    gap0_dipe1 = load_data (r'E:\python\data\gap0_dipe1.csv')
    gap0_dipe2 = load_data (r'E:\python\data\gap0_dipe2.csv')
    gap0_dipe3 = load_data (r'E:\python\data\gap0_dipe3.csv')
    gap1_dipe2 = load_data (r'E:\python\data\gap1_dipe2.csv')
    gap1_dipe3 = load_data (r'E:\python\data\gap1_dipe3.csv')
    gap2_dipe2 = load_data (r'E:\python\data\gap2_dipe2.csv')
    gap2_dipe3 = load_data (r'E:\python\data\gap2_dipe3.csv')
    gap1_dipe30 = load_data (r'E:\python\data\gap1_dipe30.csv')
    gap1_dipe31 = load_data (r'E:\python\data\gap1_dipe31.csv')
    data = numpy.concatenate ((gap0_dipe1, gap0_dipe2, gap0_dipe3, gap1_dipe2, gap1_dipe3, gap2_dipe2, gap2_dipe3,
                               gap1_dipe30, gap1_dipe31), axis=1)
    # visualization(data,label)
    # data = Threshold (data, label)
    # new_data = chooseFeature(data, label, 700)
    new, a, b = Threshold(data, label)
    df = pandas.DataFrame (new)
    df.to_csv (r'data_notsort_700.csv', header=None, index=None)
    print('end')
