# coding: utf-8
from FlaskApp.models.antioxidant.dataRead import loadModel
from FlaskApp.models.antioxidant.autoCode_model import  mixGap01
from keras.optimizers import SGD
from keras.models import Sequential
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


#############################################
###获取训练好模型的权重和中间结果输出
###########################################
pathM = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/models/model_1.json'
pathW = r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/models/modelWeights_1.h5"
model = loadModel(pathM,pathW)
print(model.layers)
print("load model success")
layers = {}
for i in range(1,4,1):   #decoder 层
    layers[i] = model.layers[i].get_weights()
'''
weights = model.get_weights()
print(weights[0].shape)   # 882*650 w矩阵
print(weights[1].shape)    # b矩阵，共650个元素
print(len(weights[1]))
print("--------------------------------")
weightPath = r'layer882_650.csv'
saveData(weightPath, layers[1][0])
print("weights-------------882-->650")
print(layers[1])
'''
####获取中间层输出，查看权重矩阵是否变化
paths = {}
paths["gap0Cancer"] = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap0anti.csv'
paths["gap0Nocancer"] = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap0nonanti.csv'
paths["gap1Cancer"] = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap1anti.csv'
paths["gap1Nocancer"] = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap1nonanti.csv'
data,label = mixGap01(0, paths)


layers = {}
for i in range(9):
    layers[i] = model.layers[i]    ##固定训练好模型的参数，查看新训练后权重是否变化
    layers[i].trainable = False

for i in [1,2,3,5,8]:  ##获取encoder层和全连接激励曾的中间结果输出
    modelPart = Sequential()
    for j in range(i+1):
        modelPart.add(layers[j])
    if i == 1:
        output_dim = 650
    elif i == 2:
        output_dim = 500
    elif i ==3 :
        output_dim = 300
    elif i == 5:
        output_dim = 50
    else:
        output_dim = 25
    sgd = SGD(lr=0.0025, decay=1e-6, momentum=0.6, nesterov=True) #lr:learning rate,as small as possible
    modelPart.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    modelPart.fit(data, np.zeros((data.shape[0],output_dim)), nb_epoch=5)
    print("部分模型训练成功...")
    hidden = modelPart.predict(data)
    print("中间层结果输出...")
    print(hidden.shape)

    print("X_tsne------------------")
    X_tsne = TSNE(learning_rate=100).fit_transform(hidden)


    #分离两类数据点
    type0_x = []
    type0_y = []
    type1_x = []
    type1_y = []
    for i in range(len(label)):
        if(label[i] == 0):
            type0_x.append(X_tsne[i][0])
            type0_y.append(X_tsne[i][1])
        if(label[i] == 1):
            type1_x.append(X_tsne[i][0])
            type1_y.append(X_tsne[i][1])
    x_min = min( [min(type0_x), min(type1_x)]) -1
    x_max = max( [max(type0_x), max(type1_x)] ) +1
    y_min = min( [min(type0_y), min(type1_y)] ) -1
    y_max = max( [max(type0_y), max(type1_y)] ) +1

    type0_x = np.array(type0_x)
    print(type0_x.shape)

    type0_y= np.array(type0_y)
    type1_x = np.array(type1_x)
    type1_y= np.array(type1_y)
    print("---------------------------------")
    '''
    plt.plot(type0_x, type0_y, 'ro',c = 'green',s=10)
    plt.plot(type1_x, type1_y, 'ro',c = 'red',s=10)
    plt.axis([x_min, x_max, y_min,y_max])
    plt.show()
    '''
    color = list()
    for x in range(type1_x.shape[0]):
        color.append([1,0.1,0.1])
    color = np.asarray(color)
    plt.scatter(type0_x, type0_y,c='grey',s=30,alpha=1,marker='s',linewidths=0.5,edgecolors="black")
    plt.scatter(type1_x, type1_y,c='white',s=30,alpha=1,marker='o',linewidths=0.5,edgecolors="black")
    savePath = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/' + str(output_dim)+ ".png"
    plt.savefig(savePath)
    plt.close()
