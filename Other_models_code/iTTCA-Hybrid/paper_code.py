# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 00:02:26 2022

@author: lvyang
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler

"""PseAAC + CTDD RF 73.60 82.79 58.67 0.428 0.783"""

index=pd.read_excel("dataset/aaindex.xlsx",index_col=("AA"))
index=index.subtract(index.min(axis=1),axis=0).divide((index.max(axis=1)-index.min(axis=1)),axis=0)#归一化
index=index.T.to_dict('list')
index["-"]=np.zeros(531)

def read_data(lines):
    seq=[]
    for i  in range(len(lines)):
        if i%2!=0:
            seq.append(lines[i][:-1])
    return seq

f=open("seqset/Train_Positive.fasta",'r')
train_lines=f.readlines()
train_seq_positive=read_data(train_lines)
f.close()

f=open("seqset/Train_Negative.fasta",'r')
train_lines=f.readlines()
train_seq_negative=read_data(train_lines)
f.close()

f=open("seqset/Test_Positive.fasta",'r')
test_lines=f.readlines()
test_seq_positive=read_data(test_lines)
f.close()

f=open("seqset/Test_Negative.fasta",'r')
test_lines=f.readlines()
test_seq_negative=read_data(test_lines)
f.close()

padding_lens= len(max(max(train_seq_positive, key=len, default=''),max(train_seq_negative, key=len, default=''),
                      max(test_seq_positive, key=len, default=''),max(test_seq_negative, key=len, default='')))


def PCP(seq,padding_lens):
    data=[]
    for i in seq:
        Tseq=[x for x in i]
        Tseq=Tseq+["-"]*(padding_lens-len(Tseq))
        data.append([index[x] for x in Tseq])
    data=np.array(data)
    data=np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))
    return data
 
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
    

       
#train_positive
#train_positive_AAC=np.array(pd.read_table(r"dataset/Train_Positive_AAC.tsv",index_col="#"))
#=np.array(pd.read_table(r"dataset/Train_Positive_DPC.tsv",index_col="#"))
train_positive_PseAAC=read_PseAAC("dataset/train_positive_pseAAC.xlsx")
train_positive_CTDD=np.array(pd.read_table(r"dataset/Train_Positive_CTDD.tsv",index_col="#"))
#train_positive_PCP=PCP(train_seq_positive,padding_lens)

#train_negative
#train_negative_AAC=np.array(pd.read_table(r"dataset/Train_Negative_AAC.tsv",index_col="Unnamed: 0"))
#train_negative_DPC=np.array(pd.read_table(r"dataset/Train_Negative_DPC.tsv",index_col="Unnamed: 0"))
train_negative_PseAAC=read_PseAAC("dataset/train_negative_pseAAC.xlsx")
train_negative_CTDD=np.array(pd.read_table(r"dataset/Train_Negative_CTDD.tsv",index_col="#"))
#train_negative_PCP=PCP(train_seq_negative,padding_lens)

#test_positive
#test_positive_AAC=np.array(pd.read_table(r"dataset/Test_Positive_AAC.tsv",index_col="Unnamed: 0"))
#test_positive_DPC=np.array(pd.read_table(r"dataset/Test_Positive_DPC.tsv",index_col="Unnamed: 0"))
test_positive_PseAAC=read_PseAAC("dataset/test_positive_pseAAC.xlsx")
test_positive_CTDD=np.array(pd.read_table(r"dataset/Test_Positive_CTDD.tsv",index_col="#"))
#test_positive_PCP=PCP(test_seq_positive,padding_lens)

#test_negative
#test_negative_AAC=np.array(pd.read_table(r"dataset/Test_Negative_AAC.tsv",index_col="Unnamed: 0"))
#test_negative_DPC=np.array(pd.read_table(r"dataset/Test_Negative_DPC.tsv",index_col="Unnamed: 0"))
test_negative_PseAAC=read_PseAAC("dataset/test_negative_pseAAC.xlsx")
test_negative_CTDD=np.array(pd.read_table(r"dataset/Test_Negative_CTDD.tsv",index_col="#"))
#test_negative_PCP=PCP(test_seq_negative,padding_lens)


train_encoding=np.concatenate(((np.concatenate((train_positive_CTDD,train_positive_PseAAC),axis=1)),
                               (np.concatenate((train_negative_CTDD,train_negative_PseAAC),axis=1))),axis=0)

train_label=np.concatenate((np.ones(train_positive_CTDD.shape[0]),np.zeros(train_negative_CTDD.shape[0])),axis=0)

test_encoding=np.concatenate(((np.concatenate((test_positive_CTDD,test_positive_PseAAC),axis=1)),
                               (np.concatenate((test_negative_CTDD,test_negative_PseAAC),axis=1))),axis=0)

test_label=np.concatenate((np.ones(test_positive_CTDD.shape[0]),np.zeros(test_negative_CTDD.shape[0])),axis=0)


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

np.savez("data2-model2-train.npz",predict=score_RF_prob,label=train_label)        
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
print("indepandent test AUC指标：%.3lf"%auc)
np.savez("data2-model2-test.npz",predict=test_predict_pro,label=test_label)        