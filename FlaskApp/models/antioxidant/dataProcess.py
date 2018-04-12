# coding: utf-8
import numpy as np
import csv
import math
import random
import copy

##############################################
#两类数据一起进行中心化 ,code:0训练过程 code = 1:测试数据
###############################################
def load_data(mode, sigmoid = False):
     if(mode == 0):
        pathCancer = r'data400/train_1-gap_cancer.csv'
        pathNocancer = r'data400/train_1-gap_noncancer.csv'
        fileCancer = file(pathCancer, 'rb')   #train_1-gap_cancer.csv
        fileNocancer = file(pathNocancer, 'rb')  #train_1-gap_noncancer.csv
     else:
        pathCancer = r'data400/test_1-gap_cancer.csv'
        pathNocancer = r'data400/test_1-gap_noncancer.csv'
        fileCancer = file(pathCancer, 'rb')   #test_1-gap_cancer.csv
        fileNocancer = file(pathNocancer, 'rb') #test_1-gap_noncancer.csv

     reader = csv.reader(fileCancer)
     readerNocancer = csv.reader(fileNocancer)

     n1 = -1
     for line in reader:
        n1 = n1 + 1
     n2 = -1
     for line in readerNocancer:
         n2 = n2 + 1

     dim = (list(line)).__len__()-1
     # n1,n2表示样本个数
     cancerArray = np.zeros((n1,dim),dtype=float)
     noCancerArray = np.zeros((n2,dim),dtype=float)

     reader = csv.reader(open(pathCancer))
     readerNocancer = csv.reader(open(pathNocancer))
     i = -1
     for line in reader:
         if(i  == -1):
             i = i+1
             continue
         cancerArray[i] = np.asarray(line[1:])
         i = i+1

     i = -1
     for line in readerNocancer:
         if(i  == -1):
             i = i+1
             continue
         noCancerArray[i] = np.asarray(line[1:])
         i = i+1

     fileCancer.close()  #读取完毕，开始中心化
     fileNocancer.close()

    #中心化训练数据（两类合并进行中心化），测试数据略过此步
     dataArray = np.concatenate((cancerArray, noCancerArray))

    #保存训练数据均值，方差
     if(mode == 0):
         dataArray, mean, var = center(dataArray, sigmoid)
         fileSaveConfi = file(r'data400/train_mean_var.csv','wb')
         print('save mean and stanVar to path: data400/train_mean_var.csv')
         writer = csv.writer(fileSaveConfi)
         writer.writerow(mean)
         writer.writerow(var)
         fileSaveConfi.close()

    #合并两类标签
     label1 = np.zeros((cancerArray.shape[0], 1))
     label2 = np.ones((noCancerArray.shape[0],1))
     label  = np.concatenate( (label1,label2) )

     return dataArray, label

#删除全为0的列
def deleColumn(dataArray):
    m, n = dataArray.shape
    sum = dataArray.sum(axis=0)  #axis = 0列和
    indexs = []
    for i in range(sum.shape[0]):
        if(sum[i] == 0):
            indexs.append(i)
    dataDele = np.delete(dataArray, np.asarray(indexs, dtype=int), axis= 1)
    print('删除了%d个特征'%( indexs.__len__() ))
    return dataDele,indexs

#加入全为0的列（在indexs处）
def addColumn(dataArray,indexs):
    indexs.sort()  #正向排序
    column = np.zeros(dataArray.shape[0])
    for index in indexs:
        dataArray = np.insert(dataArray, index, values=column, axis = 1 )

    return dataArray

   #中心化处理
def center(cancerArray):
    #去除全为0的列
    cancerArray,indexsDele = deleColumn(cancerArray)

    mean = cancerArray.sum(axis = 0)#计算均值
    N = cancerArray.shape[0]
    mean = mean/N*1.0
    meanArray = np.zeros((N, cancerArray.shape[1]),dtype=float)
    stanArray = np.ones((N,cancerArray.shape[1]), dtype= float)
    i = 0
    while( i < N ):
        meanArray[i] = mean
        i = i + 1
    cancerArray = cancerArray - mean

    var2 = ( (cancerArray*cancerArray).sum(axis=0) ) / N #计算方差
    stanVar =[math.sqrt(i) for i in list(var2)]
    i = 0
    while( i < N ):
        stanArray[i] = stanVar
        i = i + 1
    cancerArray = cancerArray/stanArray

    mean = np.asarray(mean)
    mean = mean.reshape((1, len(mean)))
    stanVar = np.asarray(stanVar)
    stanVar = stanVar.reshape((1, len(stanVar)))

    #注意:重新加入全为0的列
    if(not indexsDele.__len__() == 0):
        cancerArray = addColumn(cancerArray, indexsDele)
        mean = addColumn(mean, indexsDele)
        stanVar = addColumn(stanVar, indexsDele)

    return cancerArray, mean[0], stanVar[0]

#读取均值、标准差
def load_mean(path):
    file = open(path,'r') #path=r'graphs_model/train_mean_var.csv'
    reader = csv.reader(file)
    list = []
    for line in reader:
        list.extend(line)
    array = np.asarray(list)
    array.flatten()
    return array


def readSample(mode):
    if(mode == 0):
        pathCancer = r'data400/train_1-gap_cancer.csv'
        pathNocancer = r'data400/train_1-gap_noncancer.csv'
        fileCancer = file(pathCancer, 'rb')   #test_1-gap_cancer.csv
        fileNocancer = file(pathNocancer, 'rb')
    elif(mode == 1):
        pathCancer = r'data400/test_1-gap_cancer.csv'
        pathNocancer = r'data400/test_1-gap_noncancer.csv'
        fileCancer = file(pathCancer, 'rb')   #test_1-gap_cancer.csv
        fileNocancer = file(pathNocancer, 'rb')
    reader = csv.reader(fileCancer)
    readerNocancer = csv.reader(fileNocancer)
    n1 = -1
    for line in reader:
        n1 = n1 + 1
    n2 = -1
    for line in readerNocancer:
        n2 = n2 + 1

    dim = (list(line)).__len__()-1
     # n1,n2表示样本个数
    cancerArray = np.zeros((n1,dim),dtype=float)
    noCancerArray = np.zeros((n2,dim),dtype=float)

    reader = csv.reader(open(pathCancer))
    readerNocancer = csv.reader(open(pathNocancer))
    i = -1
    for line in reader:
         if(i  == -1):
             i = i+1
             continue
         cancerArray[i] = np.asarray(line[1:])
         i = i+1

    i = -1
    for line in readerNocancer:
         if(i  == -1):
             i = i+1
             continue
         noCancerArray[i] = np.asarray(line[1:])
         i = i+1

    fileCancer.close()  #读取完毕
    fileNocancer.close()

    return cancerArray, noCancerArray

    ##########################
    #从训练数据和测试数据中抽取测试集(两类各20个)
    ##已废弃不用
    ########################
def mixExtract():
    print('抽取新的测试集.......')

    cancerArray, noCancerArray = readSample(0)
    testCancer, testNocancer = readSample(1)
    cancerAll = np.concatenate((cancerArray, testCancer))
    nocancerAll = np.concatenate((noCancerArray, testNocancer))
    #抽取新的测试集
    cancerNew,testCancer = extract(cancerAll)
    nocancerNew, testNocancer = extract(nocancerAll)

    dataArray = np.concatenate((cancerNew, nocancerNew))
    testArray = np.concatenate((testCancer, testNocancer))
    #新标签
    label = np.concatenate(( np.zeros(( cancerNew.shape[0],1)), np.ones((nocancerNew.shape[0], 1)) ))
    testLabel = np.concatenate(( np.zeros(( testCancer.shape[0],1)), np.ones((testNocancer.shape[0], 1)) ))
    return dataArray,label, testArray, testLabel

#########################
##将样本中的正负样本分离
def split2class(data, label):
    numNegtive = int(sum(label))
    numPostive = int(len(label) - numNegtive)
    dataNegtive = np.zeros((numNegtive, data.shape[1]))
    dataPositive = np.zeros((numPostive, data.shape[1]))
    j = 0
    k = 0
    for i in range (data.shape[0]):
        if label[i] == 0:
            dataPositive[j] = data[i]
            j = j+1
        else:
            dataNegtive[k] = data[i]
            k = k+1
    return dataPositive, dataNegtive
######################
##根据预测标签分离正负样本，并返回真正标签
def splitPredClass(trainArray, trainLabel, predictLabel):
    trainLabel = trainLabel.reshape((trainLabel.shape[0],1))
    trainx_label = np.append(trainArray, trainLabel, axis=1)
    model_pos, model_neg = split2class(trainx_label,predictLabel)      ##按照预测标签分开全部训练样本中的正类
    prePos_truelabel = model_pos[ :, model_pos.shape[1]-1].transpose()    #取出判定为“正类”的真实标签
    prePos = np.delete(model_pos, -1,axis = 1)

    preNeg_truelabel = model_neg[ :, model_pos.shape[1]-1].transpose()   #判定为“负类”的真实标签，计算召回率用
    return prePos, prePos_truelabel, preNeg_truelabel

## 计算结果的准确率召回率等
def recallAcc(predictLabel, truelabel):
    TN = 0
    TP = 0
    for i in range( len(predictLabel)):
        if predictLabel[i] == 0 and (truelabel[i] == 0):
                 TP = TP + 1
        elif predictLabel[i] == 1 and (truelabel[i] == 1):
                 TN = TN + 1
    acc = (TP +TN)*1.0/len(predictLabel)
    recallPos = TP*1.0 / (len(truelabel) - sum(truelabel))
    recallNeg = TN*1.0/sum(truelabel)

    dict = { "acc" : acc, "recallPos" : recallPos, "recallNeg" : recallNeg}
    return dict


def saveMean(path, mean, stanVar):
    fileSaveConfi = file(path, 'wb')
    writer = csv.writer(fileSaveConfi)

    writer.writerow(mean)
    writer.writerow(stanVar)

    fileSaveConfi.close()

def saveData(path, DataArray):
    fileSave = open(path, 'w')
    writer = csv.writer(fileSave)

    dim = DataArray.shape[1]
    headLine = ''
    for i in range(dim):
        headLine = headLine + ', V' + str(i+1)

    writer.writerow([headLine])

    m = DataArray.shape[0]
    for i in range(m):
        index = str(i+1)
        index = '" '  +  index +  '"'
        line = [index]
        line.extend( (DataArray[i].tolist()) )
        writer.writerow(line)

    fileSave.close()


import pandas as pd
from sklearn.utils import shuffle
#############################
  # 将训练数据的样本顺序打乱
#############################
def shuffleData(data,label):
    print("打乱数据行顺序shuffle------------------------------------")
    '''
    dimension = data.shape[1]
    data_label = np.append(data, label, axis = 1)
    df = pd.DataFrame(data_label)
    df = shuffle(df)
    data_label = df.as_matrix()
    label = data_label[ : , dimension]
    data  = np.delete(data_label, np.asarray(dimension, dtype=int), axis= 1)

    '''
    data_label = np.append(data, label, axis = 1)
    numData = data.shape[0]
    dimension = data.shape[1]
    numChange = int(numData*0.8)


    for i in range(numChange):
        index1 = random.randint(0,numData-1)
        index2 = random.randint(0,numData-1)
        while(index1 == index2):
            index2 = random.randint(0,numData-1)

        temp = copy.copy(data_label[index1])
        data_label[index1] = copy.copy(data_label[index2])
        data_label[index2] = copy.copy(temp)

    label = data_label[ : , dimension]
    data  = np.delete(data_label, np.asarray(dimension, dtype=int), axis= 1)
    print('after shuffle data:%d,%d' %(data.shape[0], data.shape[1]))

    return data, label

#####################################################################################
   #n折交叉数据集分割（分为验证集和训练集）,itenum =[1,10]，代表N折中的第itenum折，返回该折的训练集和验证集(最后一列为标签),按原始样本顺序取折
   ####已废弃不用
#####################################################################################
def crossValidation(dataArray, label , itenum, n=10 , sample = "original"):
    ##当涉及逆序样本时，需进行此操作
    ##将正序和对应的逆序序列维度合并，
    if( sample == "reverse"):
        print("-----------------------------")
        print("--- 10 fold cross validation on reverse samples ---")

        dataArray_merge = np.zeros((dataArray.shape[0]/2 , dataArray.shape[1]*2))
        for i in range(dataArray.shape[0]/2):     #合并正序和逆序特征维度，返回值时需分离
            dataArray_merge[i] = np.append( dataArray[ i*2 ], dataArray[ i*2 + 1 ] )
        label = label[ 0 : label.shape[0] : 2, ]  #取偶数行的标签
        dataArray = dataArray_merge

    m = dataArray.shape[0]
    foldNum = int(m/n)


    start = (itenum-1) * foldNum
    end   =  start + foldNum

    label = label.reshape((label.shape[0],1))#合并特征和标签
    dataArray = np.asarray( np.append( dataArray, label,  axis = 1) )

    valiArray = dataArray[ start : end , :  ]                              #验证集取一折
    trainArray = dataArray[0: start , : ]
    trainArray = np.concatenate( ( trainArray, dataArray[end: m , : ]) )   #训练集取9折

    ##涉及逆序样本时，需进行此操作
    if( sample == "reverse"):
        trainLabel = trainArray[ : , -1]
        np.delete(trainArray, -1)         #删除最后一列标签
        trainArray_split = np.zeros(trainArray.shape[0]*2, trainArray.shape[1]/2)
        trainArray_split[0:trainArray.shape[0] , :] = trainArray[ : , 0 : trainArray.shape[1]/2 ]      #取顺序样本
        trainArray_split[  trainArray.shape[0] : trainArray_split.shape[0] , :] = trainArray[ : , trainArray.shape[1]/2 : trainArray.shape[1]]   #取逆序样本
        trainLabel = np.concatenate((trainLabel, trainLabel))     #对应顺序+逆序的标签

        trainArray = np.concatenate((trainArray_split, trainLabel))   #加入最后一列标签

        valiArray_split = valiArray[ : , 0 : valiArray.shape[1]/2]   #只取顺序样本
        valiArray = np.concatenate((valiArray_split, valiArray[ : , -1]))

    return trainArray, valiArray


########################################
  ##用于序列翻转后，按照指定数值（numExtract）提取特征集和验证集,
  ##验证集中正负样本各取一半
  ##已废弃不用
########################################
def extractValidation(numCancer,numNocan, numExtract, dataArray, label):  #numCancer， numNocan, dataArray:加倍后数值,numExtract:1708*0.1

    label = label.reshape((label.shape[0],1))

    dataArray = np.asarray(np.append( dataArray, label,  axis = 1))# 合并特征和标签

    #正负样本各取样一半
    indexCancer = []
    indexNocancer = []
    while( len(indexCancer) < numExtract/2 ):
        index = random.randint(0, numCancer/2-1)
        if(not indexCancer.__contains__(index)):
            indexCancer.append(index)

    while( len(indexNocancer) < numExtract/2 ):
        index = random.randint(0, numNocan/2-1)
        if(not indexNocancer.__contains__(index)):
            indexNocancer.append(index)
    ###映射到加倍后的样本Index,并合并两类Index
    indexCancer = [2 * index for index in indexCancer]
    indexNocancer = [2*index + numCancer for index in indexNocancer]
    indexNocancer = [int(x) for x in indexNocancer]

    indexs = np.concatenate( ( indexCancer, indexNocancer ) )
    indexs.sort()

    #待删除的验证样本Index和其reverse序列的Index
    deleteReverse = [x+1 for x in indexs]
    deleteReverse = np.concatenate( (deleteReverse, indexs) )
    deleteReverse.sort()

    testArray = np.zeros( (len(indexCancer)+len(indexNocancer), dataArray.shape[1]) )   #保存原始顺序序列样本
    testArray2 = np.zeros((testArray.shape[0], testArray.shape[1]))                     #保存逆序序列

    for i in range( ( len(indexCancer)+len(indexNocancer) ) ) :
        testArray[i] = dataArray[indexs[i]]       #原始顺序蛋白质序列作为测试集  VS  逆序蛋白质序列作为测试集
        testArray2[i] = dataArray[ indexs[i]+1 ]

    #axis=0 行,删除验证集样本
    trainArray = np.delete(dataArray, np.asarray(list(deleteReverse)), axis = 0)
    ##返回划分后的训练、测试集(最后一列为标签)
    return trainArray,testArray,testArray2


if(__name__ =="__main__"):
    pathAnti = r'antioxidant/aafvir.csv'
    pathNoanti = r'antioxidant/aafnonvir.csv'
    antiArray, noAntiarray = readCsv(pathAnti, pathNoanti)
    antiArray = center(antiArray)[0]
    noAntiarray = center(noAntiarray)[0]
    saveData(r"antioxidant/aafAnti.csv",antiArray)
    saveData(r"antioxidant/aafNoanti.csv",noAntiarray)


