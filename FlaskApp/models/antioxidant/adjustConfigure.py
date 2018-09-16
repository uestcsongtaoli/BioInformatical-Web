# coding: utf-8


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.optimizers import SGD
from FlaskApp.models.antioxidant.dataProcess import  shuffleData, crossValidation,splitPredClass
from keras.constraints import maxnorm
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import csv,os
import time, numpy as np
from FlaskApp.models.antioxidant.autoCode_model import mixGap01,autoEncoder

##################################################
    ##原始序列的10折交叉验证
    ## 得到模型的平均准确率和验证集平均准确率、正负类的召回率
##################################################
global maxAcc,maxRecall,maxAccItenum,maxRecallItenum   #保存所有参数训练模型的准确率或正类召回率最高值
maxAcc = 0
maxRecall = 0
maxAccItenum = -1
maxRecallItenum = -1


def trainModel_10Fold(dict_confi, data, label, writer, itenum ,autoConfigure):
    global maxAcc, maxRecall, maxAccItenum, maxRecallItenum

    print("------------------------------------------")
    print("10-fold cross validation for training... ")
    trainAccuracy = 0
    valiAccuracy = 0
    antiRecall = 0
    noantiRecall = 0
    '''
    #保存交叉验证每次的准确率、召回率信息
    valiAccString = ""
    antiRecallString = ""
    noantiRecallString = ""
    '''
    for i in range(10):
        ##提取第(i+1)折的训练集和验证集
        trainArray, valiArray = crossValidation(data, label, i+1, 10)

        trainLabel = trainArray[ :, trainArray.shape[1]-1].transpose() #提取标签
        valiLabel  =  valiArray[ :, valiArray.shape[1]-1].transpose()

        trainArray = np.delete(trainArray, -1,axis = 1) #删除最后一列标签列
        valiArray = np.delete(valiArray, -1 ,axis= 1)

        ##调用模型， 开始训练
        subMode = autoEncoder(autoConfigure, trainArray, trainLabel , weightSet=True)      #weightSet:设置正负样本权重比6:1
        if(i == 0):     ##每个参数的十折中第一次，画图
            plot = True
        else:
            plot = False
        dict_result = model_auto( 1 , trainArray, trainLabel, dict_confi, subMode , valiArray, valiLabel, itenum, plot)

        ##提取准确率、召回率数值
        trainAcc_per = dict_result["acc train"]
        valiAcc_per = dict_result["acc vali"]
        anti_recall_per = dict_result["recall anti"]
        noanti_recall_per = dict_result["recall noanti"]

        ##累加准确率、召回率
        trainAccuracy = trainAccuracy + trainAcc_per
        valiAccuracy = valiAccuracy + valiAcc_per
        antiRecall = antiRecall + anti_recall_per
        noantiRecall = noantiRecall + noanti_recall_per
        '''
        ##保存十折中每次的准确率、召回率为字符串
        valiAccString = valiAccString  + str(valiAcc_per) + "; "
        antiRecallString = antiRecallString + str(anti_recall_per) + ";"
        noantiRecallString = noantiRecallString + str(noanti_recall_per) + ";"
        '''
    ##写入note和模型的参数
    if itenum == 1:
        noteString = "10_fold cross validation ......"
        writer.writerow([noteString])
        trainString = "num of train data:" + str(trainArray.shape[0]) + "   num of validata:"  + str(valiArray.shape[0])
        writer.writerow([trainString])
        writer.writerow(" ")

    dictString =  ' '
    for element in dict_confi.items():
         dictString = dictString + element[0] +  ':' + str(element[1]) + ','
    dictString = str(itenum) + ':'+dictString
    writer.writerow( [dictString] )

    ##计算十折交叉验证的平均结果
    trainAccuracy = trainAccuracy/10
    valiAccuracy = valiAccuracy/10
    antiRecall = antiRecall/10
    noantiRecall = noantiRecall/10

    ##保存所有参数的最大召回率、最大准确率
    if(valiAccuracy > maxAcc):
        maxAcc = valiAccuracy
        maxAccItenum = itenum
    if(antiRecall > maxRecall):
        maxRecall = antiRecall
        maxRecallItenum = itenum


    ##写入训练集和测试集10折交叉验证的准确率
    accString = "**acc of train set(10-fold mean): " + str(trainAccuracy) + "     acc on validation data(mean) : "  + str(valiAccuracy)
    writer.writerow([accString] )

    recallString = "**recall of antioxident: " + str(antiRecall)  + "    recall of noanti: "  + str(noantiRecall)
    writer.writerow([recallString])
    '''
    writer.writerow([ "recall per time(anti): " + antiRecallString])
    writer.writerow(["recall per time(no_anti): " + noantiRecallString])
    '''
    writer.writerow(" ")



##########################################
###autoEncoder 的encoder + 三层全连接网络
# data,label:训练集，dict_confi:全连接参数，subMode:预训练的自编码，testArray,label:验证集， itenum,plot:画图的保存名称为itenum
#########################################
def model_auto(mode, data, label, dict_confi, subMode, testArray, testLabel, itenum, plotFlag) :
     print("encoder + 3层全连接网络训练......")


     node1 = dict_confi.get('node1')
     node2 = dict_confi.get('node2')
     drop1 = dict_confi.get('drop1')
     drop2 = dict_confi.get('drop2')
     epoch = dict_confi.get('np_epoch')

     model = Sequential()

     #加入encode层
     model.add(subMode['layer0'])
     model.add(subMode['layer1'])
     model.add(subMode['layer2'])
     model.add(subMode['layer3'])

     model.add(Dense(node1, kernel_initializer='uniform', input_dim = 300, kernel_constraint= maxnorm(3)))  # W_regularizer=l2(0.01), activity_regularizer:l2norm; W_constraint=maxnorm(3)
     model.add(Activation('tanh'))  #激活函数：relu,tanh
     model.add(Dropout(drop1))

     model.add(Dense(node2, kernel_initializer='uniform', kernel_constraint = maxnorm(3)))
     model.add(Activation('tanh'))
     model.add(Dropout(drop2))

     model.add(Dense(1, kernel_initializer='uniform'))
     model.add(Activation('sigmoid'))

     sgd = SGD(lr=0.0025, decay=1e-6, momentum=0.6, nesterov=True) #lr:learning rate,as small as possible
     model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])       #交叉熵作为损失函数

     if(mode == 0):
         print("-------------------------------------------------")
         print("use 0.9 of training set to train model...")
         dict = model.fit(data, label, epochs=int(epoch), batch_size=12,  validation_split=0.10)
                                                            #shuffle: each epoch draw samples randomly or not?shuffle= True, callbacks= [checkpointer]
         acc = dict.history.get('acc')
         val_acc = dict.history.get('val_acc')

         #result = model.evaluate(testArray, testLabel, batch_size=12, verbose= 1)
         stringReturn = 'acc of train set:'+ str(acc[-1]) + '  val_acc:' + str(val_acc[-1])
         return stringReturn

     elif(mode == 1):
         print("-------------------------------------------------")
         print("use the whole training set to train model...")
         dict = model.fit(data, label, epochs=epoch, batch_size=12, class_weight={0: 6, 1:1}, validation_data=(testArray, testLabel)) #整个训练集投入训练

         acc_train = dict.history.get('acc')

        ##画图：训练集和验证集的loss-epoch变化图,保存图
         if(plotFlag == True):


             modelSave = r"graphs_model_NoU/models/model_"+str(itenum) + ".json"
             weightPath = r"graphs_model_NoU/models/modelWeights_" +str(itenum) +".h5"
             if os.path.exists(modelSave):
                os.remove(modelSave)
                os.remove(weightPath)
             saveModel(model, modelSave, weightPath)

         ##查看测试集抗氧化蛋白的召回率
         predictLabel = model.predict_classes(testArray)
         TP = 0
         TN = 0
         for i in range(len(predictLabel)):
             if predictLabel[i] == 0 and (testLabel[i] == 0):
                 TP = TP + 1
             elif predictLabel[i] == 1 and (testLabel[i] == 1):
                 TN = TN + 1

         antiOxi_Recall = TP*1.0 / (len(testLabel) - sum(testLabel))   #抗氧化蛋白召回率
         noanti_Recall  = TN*1.0/sum(testLabel)
         vali_acc = (TP + TN)*1.0/len(testLabel)


         ######返回模型训练集和测试集准确率信息
         return_dict = {}
         return_dict["acc train"] = acc_train[-1]           #加入准确率信息
         return_dict["acc vali"]  = vali_acc
         return_dict["recall anti"] = antiOxi_Recall         #加入召回率信息
         return_dict["recall noanti"] = noanti_Recall

         return   return_dict
     else:
         dict = model.fit(data, label, epochs=epoch, batch_size=12,  sample_weight= None)
         return model, dict.history.get('acc')[-1]


#############################################
####构造网络参数(自编码+全连接)，保存模型结果
## **注意修改读取数据路径和保存数据路径
#############################################
def configureCreat():
    print('-----------------------------------------------------')
    print('runing configure creat code...')
    print (time.strftime('%Y-%m-%d-%H:%M',time.localtime(time.time())) )
    print("-----------------------------------------------------")
    #读取数据,打乱样本排序
    paths = {}
    paths["gap0Cancer"] = r'antioxidant_NoU/gap0anti.csv'
    paths["gap0Nocancer"]  = r'antioxidant_NoU/gap0nonanti.csv'
    paths["gap1Cancer"] = r'antioxidant_NoU/gap1anti.csv'
    paths["gap1Nocancer"] = r'antioxidant_NoU/gap1nonanti.csv'

    data,label = mixGap01(0, paths)  #混合两种特征，并做归一化处理，训练集的均值和标准差均保存下来
    data, label = shuffleData(data, label)

    #设定保存结果数据路径
    dict_confi = {}
    fileSave = open(r'antioxidant_NoU/result_classweight_30configure.csv', 'w')
    writer = csv.writer(fileSave)
    writer.writerow(["increase weight of positive class in encoder and whole model -->6:1"])

    #网络中遍历参数设置
    node1List = [50]
    node2List = [25]
    drop1List = [0.2]
    drop2List = [0.3]
    epochList = [15]

    itenum = 0
    for node1 in node1List:
        dict_confi['node1'] = node1
        for node2 in node2List:
            dict_confi['node2'] = node2
            for drop1 in drop1List:
                dict_confi['drop1'] = drop1
                for drop2 in drop2List:
                    dict_confi['drop2'] = drop2
                    for np_epoch in epochList:
                        dict_confi['np_epoch'] = np_epoch
                        #开始遍历参数...
                        autoConfigure = {'node1': 650, 'node2':500, 'node3': 300, 'epoch':15}

                        ##开始训练模型
                        trainModel_10Fold( dict_confi, data, label, writer, itenum+1 , autoConfigure)
                        #trainModel(dict_confi, data, label, writer, itenum + 1, subMode, data, label, data, label)
                        #cluster_train(dict_confi, data, label, writer, itenum+1,autoConfigure)

                        print('！！！！！！！！！！！！！！iterator', itenum, ' is done!')
                        itenum = itenum + 1

                        if(itenum == 1):
                            writer.writerow(["###################################################################"])
                            writer.writerow(["maxAcc: "+ str(maxAcc) + "        itenum: " + str(maxAccItenum)  ])
                            writer.writerow([ "maxAntiRecall: "+ str(maxRecall) + "     itenum: " + str(maxRecallItenum)  ])
                            fileSave.close()
                            return





if(__name__ =="__main__"):
    configureCreat()