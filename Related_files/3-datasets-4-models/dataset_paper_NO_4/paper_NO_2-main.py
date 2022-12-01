# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 00:02:26 2022

@author: lvyang
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:27:52 2022

@author: lvyang
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from libsvm.svm import *
from libsvm.svmutil import *
from collections import Counter
from sklearn.model_selection import train_test_split


def libsvm_file_read(file_path):
    train_part1,train_part2 = svm_read_problem(file_path)
    train_feature= []
    for i in range(len(train_part1)):
        Tdata=[]
        Tdata.append(train_part1[i])
        for j in range(len(train_part2[i])):
            Tdata.append(train_part2[i][j+1])
        train_feature.append(Tdata)
    return np.array(train_feature)

"""PseAAC + CTDD RF 73.60 82.79 58.67 0.428 0.783"""

def read_PseAAC(file):
    import pandas as pd 
    data=pd.read_excel(file)
    out_put_data=[]
    for i in range(int(len(data)/6)+1):
        data1=np.array(data.iloc[1+i*6]).astype("float32").reshape((1,10))
        data2=np.array(data.iloc[2+i*6]).astype("float32").reshape((1,10))
        Tdata=np.concatenate((data1,data2),axis=1)
        if i==0:
            out_put_data=Tdata
            continue
        out_put_data=np.concatenate((out_put_data,Tdata),axis=0)
        
    return out_put_data
    
def get_label(file_path):
    label=[]
    f=open(file_path,'r')
    lines=f.readlines()
    f.close()
    for i in range(len(lines)):
        if i%2==0:
            seq=[]
            seq.extend(lines[i][:-1])
            if "n" in seq:
                label.append(0)
            else:
                label.append(1)
    return np.array(label)

train_PseAAC1=read_PseAAC("train_PseAAC_1.xlsx")
train_PseAAC2=read_PseAAC("train_PseAAC_2.xlsx")
train_PseAAC=np.concatenate((train_PseAAC1,train_PseAAC2),axis=0)
train_CTDD=np.array(pd.read_table(r"train_CTDD.tsv",index_col="#"))
train_encoding=np.concatenate((train_PseAAC,train_CTDD),axis=1)
train_label=get_label("Train.FASTA")


test_PseAAC=read_PseAAC("test_PseAAC.xlsx")
test_CTDD=np.array(pd.read_table(r"test_CTDD.tsv",index_col="#"))
test_encoding=np.concatenate((test_PseAAC,test_CTDD),axis=1)
test_label=get_label("Test.FASTA")


"""归一化"""
scale=MinMaxScaler()
data_fixed=scale.fit_transform(np.concatenate((train_encoding,test_encoding),axis=0))
train_encoding=data_fixed[:train_encoding.shape[0]]
test_encoding=data_fixed[train_encoding.shape[0]:]


# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
oversample = SMOTE()
train_encoding,train_label=oversample.fit_resample(train_encoding,train_label)



"""实例化RandomForestClassifier"""

estimator = RandomForestClassifier(n_jobs=-1,random_state=0,verbose=1)

"""-------------------------------------------------------------------------"""

"""n_estimators代表随机森林中决策树的数目"""

n_estimators=np.arange(20,200,20)
max_features=np.arange(1,20,2)
param_dict={"n_estimators":n_estimators,"max_features":max_features}

"""实例化GridSearchCV"""

estimator=GridSearchCV(estimator, param_grid=param_dict,cv=5)

"""-------------------------------------------------------------------------"""

estimator.fit(train_encoding,train_label)

print("最佳估计器: \n",estimator.best_estimator_)
print("最佳参数:\n",estimator.best_params_)

best_estimator=estimator.best_estimator_

"""-----------------------------十折交叉验证---------------------------------"""
k=10
np.random.seed(1234)

num=len(train_label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))

score_RF=np.zeros(num)
score_RF_prob=np.zeros(num)

for fold in range(k):
    
    trainLabel=train_label[mode!=fold]
    testLabel=train_label[mode==fold]
    
    trainFeature=train_encoding[mode!=fold]
    testFeature=train_encoding[mode==fold]
    
    best_estimator.fit(trainFeature,trainLabel)
    score_RF[mode==fold]=best_estimator.predict(testFeature)
    score_RF_prob[mode==fold]=best_estimator.predict_proba(testFeature)[:,1]

np.savez("data4-model2-train.npz",predict=score_RF_prob,label=train_label)
print("最佳估计器: \n",estimator.best_estimator_)
print("最佳参数:\n",estimator.best_params_)
        
print("Ten fold validation acc:%.3lf"%accuracy_score(train_label,score_RF))
print("Ten fold validation precision:%.3lf"%precision_score(train_label,score_RF))
print("Ten fold validation recall:%.3lf"%recall_score(train_label,score_RF))
print("Ten fold validation MCC:%.3lf"%matthews_corrcoef(train_label,score_RF))
print("Ten fold validation AUC:%.3lf"%roc_auc_score(train_label,score_RF_prob))
"""-------------------------------------------------------------------------"""

best_estimator.fit(train_encoding,train_label)
score=best_estimator.score(test_encoding,test_label)
test_predict=best_estimator.predict(test_encoding)
print("indepandent test acc：%.3lf"%score)
print("indepandent test precision：%.3lf"%precision_score(test_label,test_predict))
print("indepandent test recall：%.3lf"%recall_score(test_label,test_predict))
print("indepandent test MCC：%.3lf"%matthews_corrcoef(test_label,test_predict))

test_predict_pro=best_estimator.predict_proba(test_encoding)[:,1]

auc=roc_auc_score(test_label,test_predict_pro)
print("indepandent test AUC：%.3lf"%auc)
np.savez("data4-model2-test.npz",predict=test_predict_pro,label=test_label)