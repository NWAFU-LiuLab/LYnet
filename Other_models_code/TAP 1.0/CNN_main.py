# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:12:29 2022

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

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=100,mode="min")

def cnn(max_len,depth,l1=16,l2=256,gamma=1e-4,lr=1e-4,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth),padding='same'))
    model.add(layers.MaxPooling1D(w2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l2(gamma)))
    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    
    """model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])"""
    
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
        s=[]
        for x in i[:-1]:
            s.extend(x)
        s=s+(21-len(i))*["-"]
        encoding.append([index[x] for x in s])
    
    if type_label==1:
        label=np.ones(len(lines))
    else:
        label=np.zeros(len(lines))
    
    return np.array(encoding),label

data_positive,label_positive=aaindex("positive.txt",1)
data_negative,label_negative=aaindex("negative.txt",0)

encoding=np.concatenate((data_positive,data_negative),axis=0)
label=np.concatenate((label_positive,label_negative),axis=0)

train_encoding,test_encoding,train_label,test_label=train_test_split(encoding,label,random_state=22,train_size=0.8)

padding_lens=20
k=10
np.random.seed(1234)

num=len(train_label)
mode1=np.arange(num/2-1)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
score_aaindex_cnn=np.zeros(num)


for fold in range(k):
    
    trainLabel=train_label[mode!=fold]
    testLabel=train_label[mode==fold]
    
    trainFeature1=train_encoding[mode!=fold]
    testFeature1=train_encoding[mode==fold]
    
    m1=cnn(padding_lens,31)
    m1.fit(trainFeature1,trainLabel,batch_size=256,epochs=10000,verbose=1,validation_data=(testFeature1,testLabel),
           shuffle=True,callbacks=[earlystop_callback])
    score_aaindex_cnn[mode==fold]=m1.predict(testFeature1).reshape(len(testFeature1))
    
np.savez('cnntrain.npz',cnn=score_aaindex_cnn,label=train_label)

model1=cnn(padding_lens,31)
model1.fit(train_encoding,train_label,batch_size=256,epochs=10000,verbose=1,validation_data=(test_encoding,test_label),
          shuffle=True,callbacks=[earlystop_callback])
test_preidct=model1.predict(test_encoding).reshape(len(test_encoding))
np.savez('cnntest.npz',cnn=test_preidct,label=test_label)


train=np.load("cnntrain.npz")
test=np.load("cnntest.npz")

fpr,tpr,_=roc_curve(train['label'],train['cnn'])
fpr1,tpr1,_=roc_curve(test['label'],test['cnn'])
lw = 1
plt.subplot(121)
plt.plot(fpr1, tpr1, color='green',lw=lw, label='CNN-aaindex-test     AUC = {:.3f}'.format(auc(fpr1,tpr1)))
plt.plot(fpr, tpr, color='red',lw=lw, label='CNN-aaindex-train-ten-cross-validation     AUC = {:.3f}'.format(auc(fpr,tpr)))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()