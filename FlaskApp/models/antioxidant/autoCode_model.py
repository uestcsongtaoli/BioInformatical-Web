# coding: utf-8
from FlaskApp.models.antioxidant.dataProcess import load_data, center
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np
from keras.layers.core import Dense
import csv
import theano

######################################
##mode=0,读取训练数据，mode=1读取测试数据;paths:dict,数据路径
##读取数据、保存数据路径自定义
###########################################
def mixGap01(mode ,paths):
    print("----------------------------------------")
    print('将数据gap0,gap1合并作为特征....')
    if(mode == 0):

        gap0Cancer = paths["gap0Cancer"]
        gap0Nocancer = paths["gap0Nocancer"]

        gap1Cancer = paths["gap1Cancer"]
        gap1Nocancer = paths["gap1Nocancer"]

    #读取gap0,gap1特征
    cancer0Array,noCancer0Array = readCsv(gap0Cancer,gap0Nocancer)
    cancer1Array,noCancer1Array = readCsv(gap1Cancer,gap1Nocancer)
    #合并gap0,gap1特征
    cancerArray = np.append(cancer0Array,cancer1Array, axis = 1 )
    noCancerArray = np.append(noCancer0Array, noCancer1Array, axis =1)

    dataAll = np.concatenate((cancerArray, noCancerArray))
    #数据归一化
    if(mode == 0):
        dataAll, mean, stanVar = center(dataAll)
        mean.flatten
        stanVar.flatten
        fileSaveConfi = open(r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/train_mean_var.csv','w')
        writer = csv.writer(fileSaveConfi)
        writer.writerow(mean.tolist())
        writer.writerow(stanVar.tolist())
        fileSaveConfi.close()

    #标签
    n1 = cancerArray.shape[0]
    n2 = noCancerArray.shape[0]
    print('cancer 样本数：%d, noCancer 样本数：%d'%(n1,n2))
    label1 = np.zeros((n1,1))           #标签   cancer:0,no_cancer:1
    label2 = np.ones((n2,1))
    label = np.concatenate( (label1, label2) )

    return dataAll, label

###读取csv文件，不做任何处理
def readCsv(pathCancer, pathNocancer):
    fileCancer = open(pathCancer)
    fileNocancer = open(pathNocancer)
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

###########################
#########简单三层自编码
####auto_config为dict形式，key包含三层的节点数和epoch数，weightSet标志是否设置正负样本权重6:1
###########################
def autoEncoder(auto_config, dataArray, label, weightSet ):
    print("------------------------------------------")
    print('简单三层自编码降维....')
    dataArray.dtype = float
    node1 = auto_config['node1']
    node2 = auto_config['node2']
    node3 = auto_config['node3']
    epoch = auto_config['epoch']

    input_dim = dataArray.shape[1]
    input_img = Input( shape=(input_dim,) )    #输入
    encoded1 = Dense(node1, activation='relu')(input_img)
    encoded2 = Dense(node2, activation='relu')(encoded1)   #三层编码网络
    encoded = Dense(node3, activation='relu')(encoded2)

    decoded = Dense(node2, activation='relu')(encoded)
    decoded = Dense(node1, activation='relu')(decoded)    #三层解码网络
    decoded = Dense( input_dim, activation='relu')(decoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')   #以矩阵间欧式距离为损失函数

    if weightSet == None:
        sampleWeight = None
    else:
        sampleWeight = np.asarray(( [( (-x+1)*5+1 ) for x in (label.tolist())] ))
        sampleWeight.reshape((sampleWeight.shape[0], 1))

    autoencoder.fit(dataArray, dataArray,
                    epochs=epoch,               #训练过程
                    batch_size=50,
                    shuffle=True,
                    sample_weight= sampleWeight)

    subModel = get_subModel(autoencoder)

    #get_weights(autoencoder)
    return subModel
###########################
#########栈式自编码
##已废弃不用
###########################
def stackEncoder(auto_config, dataArray, label):
    print('栈式自编码降维....')
    dataArray = dataArray.astype(float)

    epoch = auto_config['epoch']
    nodeList = [dataArray.shape[1]]
    nodeList.append(int(auto_config['node1']))
    nodeList.append(int(auto_config['node2']))
    nodeList.append(int(auto_config['node3']))

# Layer-wise pre-training
    trained_encoders = []
    subModel = {}
    X_train_tmp = dataArray.copy()
    for n_in, n_out in zip(nodeList[:-1], nodeList[1:]):
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
        input_img = Input( shape=(n_in,) )    #968维
        encoded = Dense(n_out, activation='relu')(input_img)
        decoded = Dense(n_in, activation='relu')(encoded)

        autoencoder = Model(input=input_img, output=decoded)
        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')   #以矩阵间欧式距离为损失函数

        autoencoder.fit(X_train_tmp, X_train_tmp,
                    nb_epoch=epoch,
                    batch_size=50,
                    shuffle=True)
        # Update training data400
        autoencoder.layers.pop() #删除decode层

        encoder = Model(input = input_img, output=encoded)
        X_train_tmp = encoder.predict(X_train_tmp)
        #Store trainined encoder
        trained_encoders.append(autoencoder.layers[1])
        if(n_in == dataArray.shape[1]):
            subModel['layer0'] = autoencoder.layers[0]  #保存最初输入层


    subModel['layer1'] = trained_encoders[0]
    subModel['layer2'] = trained_encoders[1]
    subModel['layer3'] = trained_encoders[2]

    return subModel
##以dict形式返回自编码的解码层
def get_subModel(model):
    subModel = {}

    subModel['layer0'] = model.layers[0]
    subModel['weight0'] = model.layers[0].get_weights()

    subModel['layer1'] = model.layers[1]
    subModel['weight1'] = model.layers[1].get_weights()

    subModel['layer2'] = model.layers[2]
    subModel['weight2'] = model.layers[2].get_weights()

    subModel['layer3'] = model.layers[3]
    subModel['weight3'] = model.layers[3].get_weights()

    return subModel

##################################
#########获得模型的encode 和decode参数
#################################
def get_weights(model):
    encode_650 = model.layers[1].get_weights()[0]
    #print(model.layers[1].output_dim)
    encode_500 = model.layers[2].get_weights()[0]
    encode_300 = model.layers[3].get_weights()[0]

    decode_500 = model.layers[4].get_weights()[0]
    decode_650 = model.layers[5].get_weights()[0]
    decode_968 = model.layers[6].get_weights()[0]

    compareWeight(encode_650, decode_968)
    compareWeight(encode_500, decode_650)
    compareWeight(encode_300, decode_500)

#############
##已废弃不用
def compareWeight( encode, decode ):
     inverse = np.linalg.pinv(encode) #计算伪逆
     d = np.sum( (np.multiply(decode - inverse, decode-inverse)), axis=0).sum()
     d = d/(np.sum( (np.multiply(decode, decode)), axis=0).sum())
     print( "autoEncoder:distance ration sbetween pseudo-inverse matrix and real decode matrix:", d )
     '''
     singularValue = np.linalg.eig( np.dot(encode.T , encode)  )[0]
     singularValue.tolist().sort()
     singularValue = np.asarray((singularValue))#计算特征值与向量

     singular2 = np.linalg.eig( np.dot(decode.T, decode) )
     singular2.tolist().sort()
     singular2 = np.asarray((singular2))#计算特征值与向量
     print("奇异值间欧式距离为:", np.sum(np.multiply(singularValue-singular2,singularValue-singular2), axis=0) ** 0.5)
    '''
if(__name__ =="__main__"):
    #runAutoCode()
    config_run()