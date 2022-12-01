# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:11:41 2022

@author: lvyang
"""

import numpy as np
def fun(name):
    data=np.load(name)
    label=data['label']
    predict=data['predict']
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
    print()
    
fun("data3-model2-train.npz")
fun("data3-model2-test.npz")

fun("data3-model3-train.npz")
fun("data3-model3-test.npz")

fun("data3-model4-train.npz")
fun("data3-model4-test.npz")