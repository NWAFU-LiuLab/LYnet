# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:02:34 2022

@author: lvyang
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models,layers,optimizers,regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import tensorflow as tf

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=64,mode="min")

def cnn(max_len,depth,l1=16,l2=256,gamma=1e-4,lr=1e-4,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth),padding='same'))
    model.add(layers.MaxPooling1D(w2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l2(gamma)))
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l2(gamma)))
    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    
    """model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])"""
    
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    
    return model

def rnn(max_len,depth,l1=32,l2=512,gamma=1e-4,lr=1e-4,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.LSTM(l1,activation='relu',return_sequences=True,kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l2(gamma)))
    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    return model


def aaindex(file,type_label):
    encoding=[]
    aaindex=pd.read_table('aaindex31',sep='\s+',header=None)
    aaindex=aaindex.subtract(aaindex.min(axis=1),axis=0).divide((aaindex.max(axis=1)-aaindex.min(axis=1)),axis=0)
    aa=[x for x in 'ARNDCQEGHILKMFPSTWYV']
    aaindex=aaindex.to_numpy().T
    index={x:y for x,y in zip(aa,aaindex.tolist())}
    index['-']=np.zeros(31).tolist()
    f=open(file,"r")
    lines=f.readlines()
    f.close()
    for i in lines:
        if i[0]==">":
            continue
        s=[]
        for x in i[:-1]:
            s.extend(x)
        s=s+(51-len(i))*["-"]
        encoding.append([index[x] for x in s])
    
    if type_label==1:
        label=np.ones(int(len(lines)/2))
    else:
        label=np.zeros(int(len(lines)/2))
    
    return np.array(encoding),label

train_data_positive,train_label_positive=aaindex("seqset/Train_Positive.fasta",1)
train_data_negative,train_label_negative=aaindex("seqset/Train_Negative.fasta",0)

test_data_positive,test_label_positive=aaindex("seqset/Test_Positive.fasta",1)
test_data_negative,test_label_negative=aaindex("seqset/Test_Negative.fasta",0)

train_encoding=np.concatenate((train_data_positive,train_data_negative),axis=0)
train_label=np.concatenate((train_label_positive,train_label_negative),axis=0)

row=train_encoding.shape[1]
column=train_encoding.shape[2]
train_encoding=np.reshape(train_encoding,(len(train_encoding),row*column))

# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
oversample = SMOTE()
train_encoding,train_label=oversample.fit_resample(train_encoding,train_label)

train_encoding=np.reshape(train_encoding,(len(train_encoding),row,column))



test_encoding=np.concatenate((test_data_positive,test_data_negative),axis=0)
test_label=np.concatenate((test_label_positive,test_label_negative),axis=0)

"""随机化处理"""

np.random.seed(666)
shuffle_list=[]
while True:
    i=np.random.randint(0,train_encoding.shape[0])
    if i not in shuffle_list:
        shuffle_list.append(i)
    if len(shuffle_list)==train_encoding.shape[0]:
        break

train_encoding_shuffle=train_encoding[shuffle_list]
train_label_shuffle=train_label[shuffle_list]

train_encoding=train_encoding_shuffle
train_label=train_label_shuffle

padding_lens=50

k=10
np.random.seed(1234)
num=len(train_label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
score_aaindex_cnn=np.zeros(num)
score_rnn=np.zeros(num)


for fold in range(k):
    
    trainLabel=train_label[mode!=fold]
    testLabel=train_label[mode==fold]
    
    trainFeature1=train_encoding[mode!=fold]
    testFeature1=train_encoding[mode==fold]
    
    """m1=cnn(padding_lens,31)
    m1.fit(trainFeature1,trainLabel,batch_size=256,epochs=10000,verbose=1,shuffle=True,callbacks=[earlystop_callback],
           validation_data=(testFeature1,testLabel))
    score_aaindex_cnn[mode==fold]=m1.predict(testFeature1).reshape(len(testFeature1))"""
    
    m2=rnn(padding_lens,31)
    m2.fit(trainFeature1,trainLabel,batch_size=256,epochs=10000,verbose=1,validation_data=(testFeature1,testLabel),
           shuffle=True,callbacks=[earlystop_callback])
    score_rnn[mode==fold]=m2.predict(testFeature1).reshape(len(testFeature1))
    
#np.savez('cnntrain.npz',cnn=score_aaindex_cnn,label=train_label)
np.savez('rnntrain.npz',rnn=score_rnn,label=train_label)

"""model1=cnn(padding_lens,31)
model1.fit(train_encoding,train_label,batch_size=512,epochs=10000,verbose=1,validation_data=(test_encoding,test_label),
          shuffle=True,callbacks=[earlystop_callback])
test_preidct=model1.predict(test_encoding).reshape(len(test_encoding))
np.savez('cnntest.npz',cnn=test_preidct,label=test_label)


model2=rnn(padding_lens,31)
model2.fit(train_encoding,train_label,batch_size=512,epochs=10000,verbose=1,validation_data=(test_encoding,test_label),
          shuffle=True,callbacks=[earlystop_callback])
test_preidct=model2.predict(test_encoding).reshape(len(test_encoding))
np.savez('rnntest.npz',rnn=test_preidct,label=test_label)"""


cnntrain=np.load("cnntrain.npz")
rnntrain=np.load("rnntrain.npz")
cnntest=np.load("cnntest.npz")
rnntest=np.load("rnntest.npz")

fprc,tprc,_=roc_curve(cnntrain['label'],cnntrain['cnn'])
fprr,tprr,_=roc_curve(rnntrain['label'],rnntrain['rnn'])
fprcr,tprcr,_=roc_curve(rnntrain['label'],(rnntrain['rnn']+cnntrain['cnn'])/2)
fpr1,tpr1,_=roc_curve(cnntest['label'],cnntest['cnn'])
fpr2,tpr2,_=roc_curve(rnntest['label'],rnntest['rnn'])
fpr3,tpr3,_=roc_curve(rnntest['label'],(rnntest['rnn']+cnntest['cnn'])/2)

lw = 1
plt.subplot(121)
plt.plot(fprc, tprc, color='red',lw=lw, label='CNN-train-ten-cross-validation     AUC = {:.3f}'.format(auc(fprc,tprc)))
plt.plot(fprr, tprr, color='green',lw=lw, label='RNN-train-ten-cross-validation     AUC = {:.3f}'.format(auc(fprr,tprr)))
plt.plot(fprcr, tprcr, color='black',lw=lw, label='LYNet-train-ten-cross-validation     AUC = {:.3f}'.format(auc(fprcr,tprcr)))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")


plt.subplot(122)
plt.plot(fpr1, tpr1, color='red',lw=lw, label='CNN-test     AUC = {:.3f}'.format(auc(fpr1,tpr1)))
plt.plot(fpr2, tpr2, color='green',lw=lw, label='RNN-test     AUC = {:.3f}'.format(auc(fpr2,tpr2)))
plt.plot(fpr3, tpr3, color='black',lw=lw, label='LYNet-test     AUC = {:.3f}'.format(auc(fpr3,tpr3)))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()