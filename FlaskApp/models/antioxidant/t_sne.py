# coding: utf-8
from FlaskApp.models.antioxidant.dataRead import loadModel
from FlaskApp.models.antioxidant.autoCode_model import  mixGap01
from keras.optimizers import SGD
from keras.models import Sequential
import numpy as np
from sklearn.manifold import TSNE
from sklearn.externals import joblib
from FlaskApp.models.antioxidant.dataRead import testSample_generateFeature
from sklearn import svm
from keras import backend as K
#############################################
###获取训练好模型的倒数第二层权重，经t-sne降维后，输入到SVM训练
###########################################

def result(test_data_path):
    # print('runing t-sne for the output of last 2 full connected layer')
    pathM = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/models/model_1.json'
    pathW = r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/models/modelWeights_1.h5"

    # print("predict_model1", model.predict(np.random.randint(2, size=(2, 800))))
    # print("load model success")

    ####获取中间层输出
    paths = {}
    paths["gap0Cancer"] = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap0anti.csv'
    paths["gap0Nocancer"] = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap0nonanti.csv'
    paths["gap1Cancer"] = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap1anti.csv'
    paths["gap1Nocancer"] = r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/gap1nonanti.csv'
    trainArray, label = mixGap01(0, paths)

    pathTest = test_data_path
    pathMeanVar = r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/train_mean_var.csv"
    testArray = testSample_generateFeature(pathTest, pathMeanVar)

    arrayAll = np.append(trainArray, testArray, axis=0)

    model = loadModel(pathM, pathW)

    layers = {}
    for i in range(9):  # [0,8]
        layers[i] = model.layers[i]  ##固定训练好模型的参数
        layers[i].trainable = False

    for i in [5]:  ##获取倒数第二层（8） or 第三层（5）的输出
        modelPart = Sequential()
        for j in range(i + 1):  # [0,5]层
            modelPart.add(layers[j])

    sgd = SGD(lr=0.0025, decay=1e-6, momentum=0.6, nesterov=True)  # lr:learning rate,as small as possible
    modelPart.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    modelPart.fit(trainArray, np.zeros((trainArray.shape[0], 50)), epochs=3)
    # print("部分模型训练成功...")
    hidden = modelPart.predict(arrayAll)
    # print("中间层结果输出...hidden")
    print("X_tsne------------------")
    K.clear_session()

    # X_tsne = TSNE(learning_rate=100, n_iter=1500).fit_transform(hidden)
    tsne_modle = TSNE(learning_rate=1000, n_iter=250)
    print("t_sne_modle loading")
    X_tsne = tsne_modle.fit_transform(hidden)
    print("t_sne over")

    # 训练、调参过程
    '''
    #取出所有数据(1805)经t-sne降维后的输出X_tsne，放入SVM作十折交叉验证（每次均shuffle data）
    class_weight = {0:5,1:1}
    svm_10fold(min_X_tsne, label,0.1, 2, class_weight)
    '''
    '''
    #保存所有训练数据训练的SVM模型
    label2=label.reshape([1801])
    class_weight = {0:5,1:1}
    svm = svm.SVC( C=2, gamma= 0.1,kernel = 'linear',class_weight= class_weight )
    svm.fit(X_tsne, label2)
    joblib.dump(svm,r"antioxidant_NoU/svmmodel.m")
    print("save svm model")
    '''
    hidden_testArray = np.delete(X_tsne, range(trainArray.shape[0]), axis=0)
    # 加载SVM模型
    model_svm = joblib.load(r"/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/svmmodel.m")
    pred_result = []
    pred_label = []
    for i in model_svm.predict_proba(hidden_testArray):
        pred_result.append(i)
    # for i in model_svm.predict(hidden_testArray):
    #     pred_label.append(i)
    # print(pred_label)
    # print(pred_result)
    return pred_result


if __name__ == "__main__":
    result(r'/var/www/FlaskApp/FlaskApp/models/antioxidant/antioxidant_NoU/anti.txt')