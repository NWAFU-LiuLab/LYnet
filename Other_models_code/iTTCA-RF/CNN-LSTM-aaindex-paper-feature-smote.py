# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 19:35:05 2022

@author: lvyang
"""

import pandas as pd
import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import tensorflow as tf
from libsvm.svm import *
from libsvm.svmutil import *
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

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

def GAAC(file_path):
    f=open(file_path,'r')
    lines=f.readlines()
    f.close()
    group = {
		'alphatic': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharge': 'KRH',
		'negativecharge': 'DE',
		'uncharge': 'STCPNQ'
	}

    groupKey = group.keys()

    encodings = []
    for i in range(len(lines)):
        if i%2!=0:
            sequence=lines[i][:-1]
            code = []
            count = Counter(sequence)
            myDict = {}
            for key in groupKey:
                for aa in group[key]:
                    myDict[key] = myDict.get(key, 0) + count[aa]
    
            for key in groupKey:
                code.append(myDict[key]/len(sequence))
            encodings.append(code)

    return np.array(encodings)

def AAC(file_path):
    aac_coding=[]
    f=open(file_path,'r')
    lines=f.readlines()
    f.close()
    for i in range(len(lines)):
        if i%2!=0 :
            pp=[x for x in lines[i][:-1]]
            precent=[]
            precent.append(pp.count('A')*100/len(lines[i]))
            precent.append(pp.count('R')*100/len(lines[i]))
            precent.append(pp.count('N')*100/len(lines[i]))
            precent.append(pp.count('D')*100/len(lines[i]))
            precent.append(pp.count('C')*100/len(lines[i]))
            precent.append(pp.count('Q')*100/len(lines[i]))
            precent.append(pp.count('E')*100/len(lines[i]))
            precent.append(pp.count('G')*100/len(lines[i]))
            precent.append(pp.count('H')*100/len(lines[i]))
            precent.append(pp.count('I')*100/len(lines[i]))
            precent.append(pp.count('L')*100/len(lines[i]))
            precent.append(pp.count('K')*100/len(lines[i]))
            precent.append(pp.count('M')*100/len(lines[i]))
            precent.append(pp.count('F')*100/len(lines[i]))
            precent.append(pp.count('P')*100/len(lines[i]))
            precent.append(pp.count('S')*100/len(lines[i]))
            precent.append(pp.count('T')*100/len(lines[i]))
            precent.append(pp.count('W')*100/len(lines[i]))
            precent.append(pp.count('Y')*100/len(lines[i]))
            precent.append(pp.count('V')*100/len(lines[i]))
            aac_coding.append(precent)
    return np.array(aac_coding)

#train_GPSD
train_AAC=AAC(r"seqset/Train.FASTA")
train_CTDC=libsvm_file_read(r"dataset/train_CTDC.LibSVM")
train_CTDT=libsvm_file_read(r"dataset/train_CTDT.LibSVM")
train_CTDD=libsvm_file_read(r"dataset/train_CTDD.LibSVM")
#train_GAAPC
train_GAAC=GAAC(r"seqset/Train.FASTA")
train_GDPC=libsvm_file_read(r"dataset/train_GDPC.LibSVM")
train_GTPC=libsvm_file_read(r"dataset/train_GTPC.LibSVM")
#train_ASDC
train_ASDC=libsvm_file_read(r"dataset/train_ASDC.LibSVM")
#train_PAAC
train_PAAC=libsvm_file_read(r"dataset/train_PAAC.LibSVM")

train_feature=np.concatenate((train_AAC,train_CTDC,train_CTDT,train_CTDD,train_GAAC,train_GDPC,train_GTPC,train_ASDC,train_PAAC),axis=1)
train_feature=train_feature[:,:-2]
scale=MinMaxScaler()
train_feature=scale.fit_transform(train_feature)


#test_GPSD
test_AAC=AAC(r"seqset/Test.FASTA")
test_CTDC=libsvm_file_read(r"dataset/test_CTDC.LibSVM")
test_CTDT=libsvm_file_read(r"dataset/test_CTDT.LibSVM")
test_CTDD=libsvm_file_read(r"dataset/test_CTDD.LibSVM")
#test_GAAPC
test_GAAC=GAAC(r"seqset/Test.FASTA")
test_GDPC=libsvm_file_read(r"dataset/test_GDPC.LibSVM")
test_GTPC=libsvm_file_read(r"dataset/test_GTPC.LibSVM")
#test_ASDC
test_ASDC=libsvm_file_read(r"dataset/test_ASDC.LibSVM")
#test_PAAC
test_PAAC=libsvm_file_read(r"dataset/test_PAAC.LibSVM")

test_feature=np.concatenate((test_AAC,test_CTDC,test_CTDT,test_CTDD,test_GAAC,test_GDPC,test_GTPC,test_ASDC,test_PAAC),axis=1)
test_feature=test_feature[:,:-2]
test_feature=scale.fit_transform(test_feature)


earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=100,mode="min")

aaindex=pd.read_table('aaindex31',sep='\s+',header=None)
aaindex=aaindex.subtract(aaindex.min(axis=1),axis=0).divide((aaindex.max(axis=1)-aaindex.min(axis=1)),axis=0)
aa=[x for x in 'ARNDCQEGHILKMFPSTWYV']
aaindex=aaindex.to_numpy().T
index={x:y for x,y in zip(aa,aaindex.tolist())}
index['-']=np.zeros(31).tolist()

def cnn(max_len,depth,l1=32,l2=512,gamma=1e-4,lr=1e-4,w1=3,w2=2):
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth),padding='same'))
    model.add(layers.MaxPooling1D(w2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l2(gamma)))
    #model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l2(gamma)))
    
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
row=train_encoding.shape[1]
column=train_encoding.shape[2]
train_encoding=np.reshape(train_encoding,(len(train_encoding),row*column))
train_encoding=np.concatenate((train_encoding,train_feature),axis=1)

# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
oversample = SMOTE()
train_encoding,train_label=oversample.fit_resample(train_encoding,train_label)

train_encoding=np.reshape(train_encoding,(len(train_encoding),row+28,column))

test_encoding=np.array(aminoacids_encode(test_seq, padding_lens))
test_encoding=np.concatenate((test_encoding,test_feature.reshape((len(test_feature),28,31))),axis=1)

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

k=10
np.random.seed(666)

num=len(train_label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
score_cnn=np.zeros(num)
score_rnn=np.zeros(num)

for fold in range(k):
    
    trainLabel=train_label[mode!=fold]
    testLabel=train_label[mode==fold]
    
    trainFeature1=train_encoding[mode!=fold]
    testFeature1=train_encoding[mode==fold]
    
    """m1=cnn(padding_lens+28,31)
    m1.fit(trainFeature1,trainLabel,batch_size=256,epochs=10000,verbose=1,validation_data=(testFeature1,testLabel),
           shuffle=True,callbacks=[earlystop_callback])
    score_cnn[mode==fold]=m1.predict(testFeature1).reshape(len(testFeature1))"""
    
    
    m2=rnn(padding_lens+28,31)
    m2.fit(trainFeature1,trainLabel,batch_size=256,epochs=10000,verbose=1,validation_data=(testFeature1,testLabel),
           shuffle=True,callbacks=[earlystop_callback])
    score_rnn[mode==fold]=m2.predict(testFeature1).reshape(len(testFeature1))
    
#np.savez('cnntrain.npz',cnn=score_cnn,label=train_label)
np.savez('rnntrain.npz',rnn=score_rnn,label=train_label)



"""model1=cnn(padding_lens+28,31)
model1.fit(train_encoding,train_label,batch_size=256,epochs=10000,verbose=1,validation_data=(test_encoding,test_label),
          shuffle=True,callbacks=[earlystop_callback])
test_preidct=model1.predict(test_encoding).reshape(len(test_encoding))


np.savez('cnntest.npz',cnn=test_preidct,label=test_label)"""

model2=rnn(padding_lens+28,31)
model2.fit(train_encoding,train_label,batch_size=256,epochs=10000,verbose=1,validation_data=(test_encoding,test_label),
          shuffle=True,callbacks=[earlystop_callback])
test_preidct=model2.predict(test_encoding).reshape(len(test_encoding))
np.savez('rnntest.npz',rnn=test_preidct,label=test_label)


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
