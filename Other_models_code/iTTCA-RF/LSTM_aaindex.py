# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 16:56:09 2022

@author: lvyang
"""

import pandas as pd
import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import tensorflow as tf

aaindex=pd.read_table('aaindex31',sep='\s+',header=None)
aaindex=aaindex.subtract(aaindex.min(axis=1),axis=0).divide((aaindex.max(axis=1)-aaindex.min(axis=1)),axis=0)
aa=[x for x in 'ARNDCQEGHILKMFPSTWYV']
aaindex=aaindex.to_numpy().T
index={x:y for x,y in zip(aa,aaindex.tolist())}
index['-']=np.zeros(31).tolist()

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=100,mode="min")

def LSTM(max_len,depth,l1=32,l2=512,gamma=1e-4,lr=1e-4,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.LSTM(l1,activation='tanh',return_sequences=True,kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    return model

def read_data(lines):
    label=[]
    seq=[]
    for i  in range(len(lines)):
        if i%2==0:
            temp=[]
            temp.extend(lines[i])
            if "n" in temp:
                label.append(0)
            else:
                label.append(1)
        else:
            seq.append(lines[i][:-1])
    return seq,np.array(label)

def aminoacids_encode(seq,max_len):
    encoding=[]
    for i in seq:
        s=[]
        for x in i:
            s.extend(x)
        s=s+(max_len-len(i))*["-"]
        encoding.append([index[x] for x in s])
    return encoding

f=open("seqset/Train.FASTA",'r')
train_lines=f.readlines()
train_seq,train_label=read_data(train_lines)
f.close()

f=open("seqset/Test.FASTA",'r')
test_lines=f.readlines()
test_seq,test_label=read_data(test_lines)
f.close()

padding_lens= len(max(max(train_seq, key=len, default=''),max(test_seq, key=len, default='')))

train_encoding=np.array(aminoacids_encode(train_seq, padding_lens))
test_encoding=np.array(aminoacids_encode(test_seq, padding_lens))

"""k=10
np.random.seed(1234)

num=len(train_label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
score_aaindex_LSTM=np.zeros(num)

for fold in range(k):
    
    trainLabel=train_label[mode!=fold]
    testLabel=train_label[mode==fold]
    
    trainFeature1=train_encoding[mode!=fold]
    testFeature1=train_encoding[mode==fold]
    
    m1=LSTM(padding_lens,31)
    m1.fit(trainFeature1,trainLabel,batch_size=32,epochs=250,verbose=1)
    score_aaindex_LSTM[mode==fold]=m1.predict(testFeature1).reshape(len(testFeature1))
    
np.savez('LSTMtrain.npz',LSTM=score_aaindex_LSTM,label=train_label)"""

model=LSTM(padding_lens,31)
model.fit(train_encoding,train_label,batch_size=256,epochs=10000,verbose=1,validation_data=(test_encoding,test_label),
          shuffle=True,callbacks=[earlystop_callback])
test_preidct=model.predict(test_encoding).reshape(len(test_encoding))


np.savez('LSTMtest.npz',LSTM=test_preidct,label=test_label)

train=np.load("LSTMtrain.npz")
test=np.load("LSTMtest.npz")

fpr,tpr,_=roc_curve(train['label'],train['LSTM'])
fpr1,tpr1,_=roc_curve(test['label'],test['LSTM'])
lw = 1
plt.subplot(121)
plt.plot(fpr1, tpr1, color='green',lw=lw, label='LSTM-aaindex-test     AUC = {:.3f}'.format(auc(fpr1,tpr1)))
plt.plot(fpr, tpr, color='red',lw=lw, label='LSTM-aaindex-train-ten-cross-validation     AUC = {:.3f}'.format(auc(fpr,tpr)))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
