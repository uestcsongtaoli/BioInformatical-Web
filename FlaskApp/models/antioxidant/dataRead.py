#coding: utf-8

import numpy as np
import csv

##全局变量
lettersAll = set()
##读取氨基酸序列中的所有氨基酸种类,返回所有字母和提取的氨基酸序列
def readLetter(path):
    file = open(path, 'r')
    letters = set()
    lines = file.readlines()
    lines_extract = []
    for line in lines:
        if not line.startswith(">"):
            letters.update(line.strip())
            lines_extract.append(line.strip())

    lettersAll.update(letters)
    return  letters,lines_extract

def generate_feature(gap, lines_extract, features_name):

    ##根据序列，计算特征向量
    m = len(lines_extract)
    n = len(features_name)
    feature_matrix = np.zeros((m,n))
    i = 0
    for line in lines_extract:
        vector = feature_matrix[i]
        length = len(line)
        for j in range(0,length-gap-1):
            index = features_name.index((line[j] + line[j + gap + 1]))
            vector[ index ] = vector[ index ] + 1
        vector = vector/(length-gap-1)
        feature_matrix[i] = vector

        i = i+1
    return feature_matrix

##此处paths为dict，传入训练集和测试集的路径
def saveDataInfo(paths):
    cancerPath = paths["positive"]
    nocancerPath = paths["negtive"]

    pathSave =  cancerPath[0:cancerPath.index(".")]+ r"_dataInformation.csv"  #保存数据集中提取的信息
    fileSave = open(pathSave, 'w')
    fileSave.write('')
    writer = csv.writer(fileSave)

    if  not paths["testPosi"]== "empty":
        testPosi = paths["testPosi"]
        testNeg  = paths["testNeg"]
        test_letterPos,test_sequePos = readLetter(testPosi)
        test_lettersNeg,test_sequeNeg = readLetter(testNeg)
        writer.writerow(["num test: "  + str(len(test_sequePos)) + "  " + str(len(test_sequeNeg))])

    letterPos,sequePos = readLetter(cancerPath)
    lettersNeg,sequeNeg = readLetter(nocancerPath)
    lettersAll.update(letterPos)
    lettersAll.update(lettersNeg)
    numPos = len(sequePos)
    numNeg = len(sequeNeg)
    writer.writerow(["numTrain: " + str(numPos+numNeg)+ ", " + str(numPos) + "(pos)  " + str(numNeg)  + "(neg)"])
    writer.writerow(["num amino acid: " + str(len(lettersAll)) + ",  "+str(len(letterPos)) + "   " + str(len(lettersNeg))])
    writer.writerow([lettersAll])

def readFeature(path):  #读取400维特征向量的名称
    file = open(path, "r")
    line = file.readlines()[0].strip()
    features = line.split(",")
    return features

import os
from FlaskApp.models.antioxidant.dataProcess import load_mean
#读取测试序列数据，归一化后并生成特征矩阵
def testSample_generateFeature(path_test,path_meanVar):
    set20={"R", "D", "E", "N", "Q", "K", "H", "L", "I", "V", "A", "M", "F", "S", "Y", "T", "W", "P", "G", "C"}
    path_feature = r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/colname.csv"
    feature_names=readFeature(path_feature)
    letters,lines_extract_test=readLetter(path_test)

    if(letters-set20):#20种字母以外的字母,则退出程序
        print("test sample contains Illegal letter!")
        os._exit(1)
    gap0Feature=generate_feature(0,lines_extract_test,feature_names)
    gap1Feature=generate_feature(1,lines_extract_test,feature_names)
    testdata = np.append(gap0Feature, gap1Feature, axis = 1)  #合并gap0,gap1特征
    mean_var=load_mean(path_meanVar)
    testArray = centerTest(mean_var, testdata)  #归一化后的数据
    return testArray


def centerTest(confiArray, dataArray):
    mean = confiArray[0]
    stanVar = confiArray[1]

    M = dataArray.shape[0]
    N = dataArray.shape[1]

    meanCancerArray = np.zeros((M, N),dtype=float)
    i = 0
    while(i < M):
        meanCancerArray[i] = float(mean)
        i = i + 1

    dataArray = dataArray - meanCancerArray
    stanArray = np.ones((M,N), dtype=float)
    i = 0
    while( i < M ):
        stanArray[i] = float(stanVar)
        i = i + 1
    dataArray = dataArray/stanArray

    return dataArray

##加载模型
from keras.models import model_from_json
from keras.optimizers import SGD
def loadModel(pathModel, pathw):
    model = model_from_json(open(pathModel).read())
    model.load_weights(pathw)
    sgd = SGD(lr=0.0025, decay=1e-6, momentum=0.6, nesterov=True) #lr:learning rate,as small as possible
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


from FlaskApp.models.antioxidant.dataProcess import saveData
if(__name__ =="__main__"):
    path={}
    path['positive'] = r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/anti.txt"
    path['negtive']  = r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/nonanti.txt"
    path['testPosi'] = "/var/www/FlaskApp/input_data/antioxidant.txt"
    saveDataInfo(path)

    path = r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/colname.csv"
    feature_names=readFeature(path)
    #读取正负样本序列，并生成gap0,gap1特征矩阵
    letters,lines_extract_positive=readLetter(r"antioxidant_NoU/anti.txt")
    letters,lines_extract_negative=readLetter(r"antioxidant_NoU/nonanti.txt")
    gap0anti=generate_feature(0,lines_extract_positive,feature_names)
    gap1anti=generate_feature(1,lines_extract_positive,feature_names)
    gap0Noanti=generate_feature(0,lines_extract_negative,feature_names)
    gap1Noanti=generate_feature(1,lines_extract_negative,feature_names)
    #保存特征矩阵
    saveData(r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap0anti.csv",gap0anti)
    saveData(r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap1anti.csv",gap1anti)
    saveData(r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap0nonanti.csv",gap0Noanti)
    saveData(r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap1nonanti.csv",gap1Noanti)

if(__name__ =="__main__"):
    path=r"C:\Users\wxz\PycharmProjects\python3.0\antioxidant_NoU\test.txt"
    path2=r"C:\Users\wxz\PycharmProjects\python3.0\antioxidant_NoU\train_mean_var.csv"
    testSample_generateFeature(path,path2)





