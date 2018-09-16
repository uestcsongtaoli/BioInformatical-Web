# -*- coding:utf-8 -*-

from sklearn.externals import joblib

from FlaskApp.models.bacteriophage import dataprocess
from FlaskApp.models.bacteriophage.Threshold8_3 import *
from FlaskApp.models.bacteriophage.dataprocess import *

pyplot.style.use('ggplot')

positive = 99
SEED = 123
numpy.random.seed(SEED)

label = [1 if i < positive else 0 for i in range (307)]

def load_data(path):

    dataframe = pandas.read_csv(path, header=None, index_col=None)
    df = dataframe.replace(0, numpy.nan)
    df.dropna(axis=1, how='all')
    df = df.replace(numpy.nan, 0)
    data = df.values

    return data


def shuffleData(data):
    #打乱顺序

    if len(data[0]) > 0:
        idx = numpy.arange(len(data[0]))
        numpy.random.shuffle(idx)
        idx = idx[:len(data[0])]
        data = ([data[0][n] for n in idx], [data[1][n] for n in idx])

    return numpy.array(data[0]), numpy.array(data[1])


def chooseFeature(data, label):

    skf = SelectKBest(f_classif, 6700)
    skf = skf.fit(data, label)
    data_new = skf.transform(data)

    return data_new, skf


def show(true, pred):

    print('auROC score：%f' % metrics.roc_auc_score (true, pred))
    print('Accuracy :%f' % metrics.accuracy_score (true, pred))
    print('classification_report:\n', metrics.classification_report(true, pred))
    print('confusion matrix\n', metrics.confusion_matrix(true, pred))


def build_model(train_data_path):

    # params = {'alpha': [0.01, 0.1, 1]}
    # grid = GridSearchCV(MultinomialNB(fit_prior=False), param_grid=params, n_jobs=-1)
    # grid.fit(data, label)
    #
    # print('train data:', data.shape)
    # clf = MultinomialNB(grid.best_params_['alpha'], fit_prior=False)
    # clf.fit(data, label)
    # joblib.dump(clf, train_data_path + "train_model.m")

    clf = joblib.load(train_data_path + "train_model.m")

    return clf


def train_fit(data, label):


    # 根据训练集选定的阈值更新测试集
    train_x, thre, idx_score = Threshold(data, label)
    data = pandas.DataFrame(train_x)
    data.to_csv(r'E:\python\phage\代码整理\phage\data\train_data.csv', header=None, index=None)

    model = build_model(train_x, label)

    return thre, idx_score, model


def predict(test, threshold, idx_score, model):

    test_x = update_threshold(test, threshold, idx_score)
    test_pred = model.predict_proba(test_x)

    return test_pred

def train_test(train_data_path,  test_data_path, test_save_data_path, label=label):

    # gap0_dipe1 = load_data (train_data_path + r'gap0_dipe1.csv')
    # gap0_dipe2 = load_data (train_data_path + r'gap0_dipe2.csv')
    # gap0_dipe3 = load_data (train_data_path + r'gap0_dipe3.csv')
    # gap1_dipe2 = load_data (train_data_path + r'gap1_dipe2.csv')
    # gap1_dipe3 = load_data (train_data_path + r'gap2_dipe31.csv')
    # gap2_dipe2 = load_data (train_data_path + r'gap2_dipe2.csv')
    # gap2_dipe30 = load_data (train_data_path + r'gap2_dipe32.csv')
    # gap1_dipe30 = load_data (train_data_path + r'gap1_dipe32.csv')
    # gap1_dipe31 = load_data (train_data_path + r'gap1_dipe31.csv')
    # gap2_dipe33 = load_data (train_data_path + r'gap2_dipe33.csv')

    # data = numpy.concatenate((gap0_dipe1, gap0_dipe2, gap0_dipe3, gap1_dipe2, gap1_dipe3, gap2_dipe2, gap2_dipe30,
    #                                 gap1_dipe30, gap1_dipe31, gap2_dipe33), axis=1)

    # data = load_data(train_data_path + r'train_data.csv')
    # data, label = shuffleData((data, label))
    # new_data, skf = chooseFeature(data, label)
    # threshold, idx_score, model = train_fit(new_data, label)
    # numpy.savez(train_data_path + "params.npz", skf.get_support(), threshold, idx_score)


    npzfile = numpy.load(train_data_path + "/params.npz")
    threshold, idx_score = npzfile['arr_1'], npzfile['arr_2']
    model = build_model(train_data_path)

    # ***缺test路径
    test_sequence = dataprocess.load_data(test_data_path)
    test_data = dataprocess.main(test_sequence, test_save_data_path)

    # gap0_dipe1 = load_data(test_save_data_path + r'gap0_dipe1.csv')
    # gap0_dipe2 = load_data(test_save_data_path + r'gap0_dipe2.csv')
    # gap0_dipe3 = load_data(test_save_data_path + r'gap0_dipe3.csv')
    # gap1_dipe2 = load_data(test_save_data_path + r'gap1_dipe2.csv')
    # gap1_dipe3 = load_data(test_save_data_path + r'gap2_dipe31.csv')
    # gap2_dipe2 = load_data(test_save_data_path + r'gap2_dipe2.csv')
    # gap2_dipe30 = load_data(test_save_data_path + r'gap2_dipe32.csv')
    # gap1_dipe30 = load_data(test_save_data_path + r'gap1_dipe32.csv')
    # gap1_dipe31 = load_data(test_save_data_path + r'gap1_dipe31.csv')
    # gap2_dipe33 = load_data(test_save_data_path + r'gap2_dipe33.csv')
    # test_data = numpy.concatenate((gap0_dipe1, gap0_dipe2, gap0_dipe3, gap1_dipe2, gap1_dipe3, gap2_dipe2, gap2_dipe30,
    #                            gap1_dipe30, gap1_dipe31, gap2_dipe33), axis=1)
    # test_data = load_data(test_save_data_path + 'test_data.csv')
    test_data = test_data[:, npzfile['arr_0']]
    return predict(test_data, threshold, idx_score, model)

if __name__ == '__main__':

    train_data_path = r'E:\python\phage\代码整理\phage\data\\' #'训练集csv文件地址'
    test_data_path = r'E:\python\data\shao\\anti.txt'#'测试集txt文件地址'
    test_save_data_path = r'E:\python\phage\代码整理\phage\testdata\\'#'输入一个地址保存测试集csv文件'
    result = train_test(train_data_path,  test_data_path, test_save_data_path, label=label)
    print(result)