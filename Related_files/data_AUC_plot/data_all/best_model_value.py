# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 21:31:56 2022

@author: lvyang
"""
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np

model2_train=np.load("data_all-model2-train.npz")
model2_test=np.load("data_all-model2-test.npz")
model3_train=np.load("data_all-model3-train.npz")
model3_test=np.load("data_all-model3-test.npz")
model4_train=np.load("data_all-model4-train.npz")
model4_test=np.load("data_all-model4-test.npz")

mfpr20,mtpr20,_=roc_curve(model2_train['label'],model2_train['predict'])
mfpr21,mtpr21,_=roc_curve(model2_test['label'],model2_test['predict'])
mfpr30,mtpr30,_=roc_curve(model3_train['label'],model3_train['predict'])
mfpr31,mtpr31,_=roc_curve(model3_test['label'],model3_test['predict'])
mfpr40,mtpr40,_=roc_curve(model4_train['label'],model4_train['predict'])
mfpr41,mtpr41,_=roc_curve(model4_test['label'],model4_test['predict'])


cnntrain=np.load("data_allcnntrain.npz")
rnntrain=np.load("data_allrnntrain.npz")
cnntest=np.load("data_allcnntest.npz")
rnntest=np.load("data_allrnntest.npz")

fprc,tprc,_=roc_curve(cnntrain['label'],cnntrain['cnn'])
fprr,tprr,_=roc_curve(rnntrain['label'],rnntrain['rnn'])
fpr1,tpr1,_=roc_curve(cnntest['label'],cnntest['cnn'])
fpr2,tpr2,_=roc_curve(rnntest['label'],rnntest['rnn'])

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
fprcr,tprcr,_=roc_curve(rnntrain['label'],rnntrain['rnn']*order1*min_step+cnntrain['cnn']*(1-order1*min_step))
fpr3,tpr3,_=roc_curve(rnntest['label'],rnntest['rnn']*order2*min_step+cnntest['cnn']*(1-order2*min_step))


lw = 1
plt.figure(figsize=(10,5.2),dpi=1000)
plt.subplot(121)
plt.plot(mfpr20,mtpr20, color='red',lw=lw, label='iTTCA-Hybrid     AUC = {:.3f}'.format(auc(mfpr20,mtpr20)))
plt.plot(mfpr30,mtpr30, color='green',lw=lw, label='TAP 1.0     AUC = {:.3f}'.format(auc(mfpr30,mtpr30)))
plt.plot(mfpr40,mtpr40, color='blue',lw=lw, label='iTTCA-RF     AUC = {:.3f}'.format(auc(mfpr40,mtpr40)))
plt.plot(fprcr, tprcr, color='black',lw=lw, label='LYNet     AUC = {:.3f}'.format(auc(fprcr,tprcr)))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")


plt.subplot(122)
plt.plot(mfpr21,mtpr21, color='red',lw=lw, label='iTTCA-Hybrid     AUC = {:.3f}'.format(auc(mfpr21,mtpr21)))
plt.plot(mfpr31,mtpr31, color='green',lw=lw, label='TAP 1.0    AUC = {:.3f}'.format(auc(mfpr31,mtpr31)))
plt.plot(mfpr41,mtpr41, color='blue',lw=lw, label='iTTCA-RF     AUC = {:.3f}'.format(auc(mfpr41,mtpr41)))
plt.plot(fpr3, tpr3, color='black',lw=lw, label='LYNet     AUC = {:.3f}'.format(auc(fpr3,tpr3)))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()