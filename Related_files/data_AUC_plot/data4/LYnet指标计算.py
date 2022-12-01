# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:36:22 2022

@author: lvyang
"""
import numpy as np
from sklearn.metrics import roc_curve,auc,matthews_corrcoef

cnntrain=np.load("data4cnntrain.npz")
rnntrain=np.load("data4rnntrain.npz")
cnntest=np.load("data4cnntest.npz")
rnntest=np.load("data4rnntest.npz")
min_step=1e-3

auc_list1=[]
auc_list2=[]
for i in np.arange(0,1+min_step,min_step):
    fprcr,tprcr,_=roc_curve(rnntrain['label'],rnntrain['rnn']*i+cnntrain['cnn']*(1-i))
    auc_list1.append(auc(fprcr,tprcr))
    fpr3,tpr3,_=roc_curve(rnntest['label'],rnntest['rnn']*i+cnntest['cnn']*(1-i))
    auc_list2.append(auc(fpr3,tpr3))
order1=auc_list1.index(max(auc_list1))
order2=auc_list2.index(max(auc_list2))

train_label=cnntrain["label"]
train_predict=rnntrain['rnn']*order1*min_step+cnntrain['cnn']*(1-order1*min_step)

test_label=cnntest["label"]
test_predict=rnntest['rnn']*order2*min_step+cnntest['cnn']*(1-order2*min_step)

def fun(label,predict):
    predict_label=[]
    for i in predict:
        if i>=0.5:
            predict_label.append(1)
        else:
            predict_label.append(0)
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(label)):
        if label[i]==1 and predict_label[i]==1:
            TP+=1
        elif label[i]==0 and predict_label[i]==0:
            TN+=1
        elif label[i]==0 and predict_label[i]==1:
            FN+=1
        elif label[i]==1 and predict_label[i]==0:
            FP+=1
    
    print("ACC=",(TP+TN)/(TP+TN+FP+FN))
    print("p=",TP/(TP+FP))
    print("sn=",TP/(TP+FN))
    print("sp=",TN/(TN+FP))
    print("MCC=",matthews_corrcoef(label,predict_label))
    print()
    
fun(train_label,train_predict)
fun(test_label,test_predict)