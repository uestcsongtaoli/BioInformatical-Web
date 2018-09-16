import csv
from sklearn.externals import joblib
import re
import itertools as iters
import numpy as np
import pandas as pd




def test(matrix):
    with open(matrix, "r") as file:
        train_fdata = [line.strip() for line in file if '>' != line[0]]

    SAA = ('ACDEFGHIKLMNPQRSTVWY')
    DIPEPTIDE = []

    for dipeptide in iters.product(SAA, repeat=3):
        DIPEPTIDE.append(''.join(dipeptide))

    df = pd.read_csv(r'/var/www/FlaskApp/FlaskApp/models/lda/800features.csv',index_col = 0)
    cols = list(df.iloc[:,0])
    
    matrix = np.zeros((len(train_fdata),800))
    for j in range(len(cols)):
        gap1 = int(cols[j][1])
        gap2 = int(cols[j][2])
        tripeptide = cols[j][0] + cols[j][3] + cols[j][6]
        for i in range(len(train_fdata)):
            protein = train_fdata[i]
            n = 0
            loops = len(train_fdata[i])- int(gap1) - int(gap2) -2
            for start in range(loops):
                
                dipeptide = protein[start] + protein[start + gap1 + 1] + protein[start + 2 + int(gap1) + int(gap2)]

                if dipeptide == tripeptide:
                    n += 1
                    
            matrix[i,j] = n/loops
    matrix=pd.DataFrame(matrix,columns = cols)
    # matrix.to_csv('first800dimension.csv')
    # matrix= pd.read_csv('first800dimension.csv',index_col = 0)
    
    #6次降维
    for number in range(6):
        filename = '%s%d%s' % ('/var/www/FlaskApp/FlaskApp/models/lda/order/lda_order__', number, '.csv')        #这里读取的是每一次迭代记录的降维的特征
        reader = csv.reader(open(filename, 'r', errors='ignore'))
        transform_test_x = []       #用来把两两降维的特征合在一起再转化为新的降维后的矩阵
        for items in reader:
            for item in items:
                order=re.findall('\d+',item)
                first=int(order[0])
                try:
                    second=int(order[1])
                    LDA=joblib.load('%s%d%s%d%s%d%s' %('/var/www/FlaskApp/FlaskApp/models/lda/model/LDA__',number,'_',first,'_',second,'.model'))   #加载模型两两降维
                    feature=LDA.transform(matrix.iloc[:,[first,second]])
                    c=[]
                    for x in feature:  # 将嵌套的list平整为一个list，方便转化为矩阵
                        c.extend(x)
                    transform_test_x.append(c)
                except:
                    feature=np.array(matrix.iloc[:,[first]])
                    c = []
                    for x in feature:  # 将嵌套的list平整为一个list，方便转化为矩阵
                        c.extend(x)
                    transform_test_x.append(c)


        transform_matrix = pd.DataFrame(transform_test_x).T   #降维后的特征形成新的矩阵进行下一次降维
        # path ='%s%d%s' %('RawData/Cancerlectin/model_test_',number,'.csv')       #保存每一次降维后的数据
        # transform_matrix.to_csv(path)
        matrix=transform_matrix



    #标准化
    scaler=joblib.load('/var/www/FlaskApp/FlaskApp/models/lda/model/scaler.model')
    matrix=np.array(matrix)
    matrix=scaler.transform(matrix)

    #SVM
    clf=joblib.load('/var/www/FlaskApp/FlaskApp/models/lda/model/svm.model')
    score=clf.predict_proba(matrix)

    return score


def main():
    a = 'non-cancerlectin.txt'
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    score=test(a)
    print(score)

if __name__ == '__main__':
    main()


