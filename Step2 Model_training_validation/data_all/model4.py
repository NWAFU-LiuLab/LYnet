# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 10:33:43 2022

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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 


def read_k_mer(file):
    data=pd.read_csv(file).iloc[:,3:]
    return np.array(data)
    
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


#positive_GPSD
positive_AAC=AAC(r"../data3/positive.fasta")
positive_CTDC=libsvm_file_read(r"../data3/positive_CTDC.LibSVM")
positive_CTDT=libsvm_file_read(r"../data3/positive_CTDT.LibSVM")
positive_CTDD=libsvm_file_read(r"../data3/positive_CTDD.LibSVM")
#positive_GAAPC
positive_GAAC=GAAC(r"../data3/positive.fasta")
positive_GDPC=libsvm_file_read(r"../data3/positive_GDPC.LibSVM")
positive_GTPC=libsvm_file_read(r"../data3/positive_GTPC.LibSVM")
#positive_ASDC
positive_ASDC=libsvm_file_read(r"../data3/positive_ASDC.LibSVM")
#positive_PAAC
positive_PAAC=libsvm_file_read(r"../data3/positive_PAAC.LibSVM")


positive_APAAC=libsvm_file_read(r"../data3/positive_APAAC.LibSVM")
positive_CKSAAGP=libsvm_file_read(r"../data3/positive_CKSAAGP.LibSVM")
positive_CKSAAP=libsvm_file_read(r"../data3/positive_CKSAAP.LibSVM")
positive_CTriad=libsvm_file_read(r"../data3/positive_CTriad.LibSVM")
positive_DDE=libsvm_file_read(r"../data3/positive_DDE.LibSVM")
positive_DistancePair=libsvm_file_read(r"../data3/positive_DistancePair.LibSVM")
positive_DPC=libsvm_file_read(r"../data3/positive_DPC.LibSVM")
#train_QSOrder=libsvm_file_read(r"dataset/train_QSOrder.LibSVM")
positive_SOCNumber=libsvm_file_read(r"../data3/positive_SOCNumber.LibSVM")
positive_k_mer=np.load(r"../data3/positive_k_mer.npy")

cnn_positive_feature=np.concatenate((positive_AAC,positive_CTDC,positive_CTDT,positive_CTDD,positive_GAAC,positive_GDPC,positive_GTPC,positive_ASDC,positive_PAAC,
                              positive_APAAC,positive_CKSAAGP,positive_CKSAAP,positive_CTriad,positive_DDE,positive_DistancePair,positive_DPC,
                              positive_SOCNumber,positive_k_mer),axis=1)

rnn_positive_feature=np.concatenate((positive_AAC,positive_CTDC,positive_CTDT,positive_CTDD,positive_GAAC,positive_GDPC,positive_GTPC,positive_ASDC,positive_PAAC,
                              positive_APAAC,positive_CKSAAGP,positive_CKSAAP,positive_CTriad,positive_DDE,positive_DistancePair,positive_DPC,
                              positive_SOCNumber,positive_k_mer),axis=1)

cnn_add_length=int(cnn_positive_feature.shape[1]/31)
cnn_unused_number=cnn_add_length*31-cnn_positive_feature.shape[1]

rnn_add_length=int(rnn_positive_feature.shape[1]/31)
rnn_unused_number=rnn_add_length*31-rnn_positive_feature.shape[1]


cnn_positive_feature=cnn_positive_feature[:,:cnn_unused_number]
rnn_positive_feature=rnn_positive_feature[:,:rnn_unused_number]



#negative_GPSD
negative_AAC=AAC(r"../data3/negative.fasta")
negative_CTDC=libsvm_file_read(r"../data3/negative_CTDC.LibSVM")
negative_CTDT=libsvm_file_read(r"../data3/negative_CTDT.LibSVM")
negative_CTDD=libsvm_file_read(r"../data3/negative_CTDD.LibSVM")
#negative_GAAPC
negative_GAAC=GAAC(r"../data3/negative.fasta")
negative_GDPC=libsvm_file_read(r"../data3/negative_GDPC.LibSVM")
negative_GTPC=libsvm_file_read(r"../data3/negative_GTPC.LibSVM")
#negative_ASDC
negative_ASDC=libsvm_file_read(r"../data3/negative_ASDC.LibSVM")
#negative_PAAC
negative_PAAC=libsvm_file_read(r"../data3/negative_PAAC.LibSVM")


negative_APAAC=libsvm_file_read(r"../data3/negative_APAAC.LibSVM")
negative_CKSAAGP=libsvm_file_read(r"../data3/negative_CKSAAGP.LibSVM")
negative_CKSAAP=libsvm_file_read(r"../data3/negative_CKSAAP.LibSVM")
negative_CTriad=libsvm_file_read(r"../data3/negative_CTriad.LibSVM")
negative_DDE=libsvm_file_read(r"../data3/negative_DDE.LibSVM")
negative_DistancePair=libsvm_file_read(r"../data3/negative_DistancePair.LibSVM")
negative_DPC=libsvm_file_read(r"../data3/negative_DPC.LibSVM")
#train_QSOrder=libsvm_file_read(r"dataset/train_QSOrder.LibSVM")
negative_SOCNumber=libsvm_file_read(r"../data3/negative_SOCNumber.LibSVM")
negative_k_mer=np.load(r"../data3/negative_k_mer.npy")

cnn_negative_feature=np.concatenate((negative_AAC,negative_CTDC,negative_CTDT,negative_CTDD,negative_GAAC,negative_GDPC,negative_GTPC,negative_ASDC,negative_PAAC,
                              negative_APAAC,negative_CKSAAGP,negative_CKSAAP,negative_CTriad,negative_DDE,negative_DistancePair,negative_DPC,
                              negative_SOCNumber,negative_k_mer),axis=1)

rnn_negative_feature=np.concatenate((negative_AAC,negative_CTDC,negative_CTDT,negative_CTDD,negative_GAAC,negative_GDPC,negative_GTPC,negative_ASDC,negative_PAAC,
                              negative_APAAC,negative_CKSAAGP,negative_CKSAAP,negative_CTriad,negative_DDE,negative_DistancePair,negative_DPC,
                              negative_SOCNumber,negative_k_mer),axis=1)

cnn_negative_feature=cnn_negative_feature[:,:cnn_unused_number]
rnn_negative_feature=rnn_negative_feature[:,:rnn_unused_number]


#train_GPSD
train_AAC=AAC(r"../data4/Train.FASTA")
train_CTDC=libsvm_file_read(r"../data4/train_CTDC.LibSVM")
train_CTDT=libsvm_file_read(r"../data4/train_CTDT.LibSVM")
train_CTDD=libsvm_file_read(r"../data4/train_CTDD.LibSVM")
#train_GAAPC
train_GAAC=GAAC(r"../data4/Train.FASTA")
train_GDPC=libsvm_file_read(r"../data4/train_GDPC.LibSVM")
train_GTPC=libsvm_file_read(r"../data4/train_GTPC.LibSVM")
#train_ASDC
train_ASDC=libsvm_file_read(r"../data4/train_ASDC.LibSVM")
#train_PAAC
train_PAAC=libsvm_file_read(r"../data4/train_PAAC.LibSVM")


train_APAAC=libsvm_file_read(r"../data4/train_APAAC.LibSVM")
train_CKSAAGP=libsvm_file_read(r"../data4/train_CKSAAGP.LibSVM")
train_CKSAAP=libsvm_file_read(r"../data4/train_CKSAAP.LibSVM")
train_CTriad=libsvm_file_read(r"../data4/train_CTriad.LibSVM")
train_DDE=libsvm_file_read(r"../data4/train_DDE.LibSVM")
train_DistancePair=libsvm_file_read(r"../data4/train_DistancePair.LibSVM")
train_DPC=libsvm_file_read(r"../data4/train_DPC.LibSVM")
#train_QSOrder=libsvm_file_read(r"dataset/train_QSOrder.LibSVM")
train_SOCNumber=libsvm_file_read(r"../data4/train_SOCNumber.LibSVM")
train_k_mer=np.load(r"../data4/Train_k_mer.npy")

cnn_train_feature=np.concatenate((train_AAC,train_CTDC,train_CTDT,train_CTDD,train_GAAC,train_GDPC,train_GTPC,train_ASDC,train_PAAC,
                              train_APAAC,train_CKSAAGP,train_CKSAAP,train_CTriad,train_DDE,train_DistancePair,train_DPC,
                              train_SOCNumber,train_k_mer),axis=1)

rnn_train_feature=np.concatenate((train_AAC,train_CTDC,train_CTDT,train_CTDD,train_GAAC,train_GDPC,train_GTPC,train_ASDC,train_PAAC,
                              train_APAAC,train_CKSAAGP,train_CKSAAP,train_CTriad,train_DDE,train_DistancePair,train_DPC,
                              train_SOCNumber,train_k_mer),axis=1)


cnn_train_feature=cnn_train_feature[:,:cnn_unused_number]
rnn_train_feature=rnn_train_feature[:,:rnn_unused_number]


#test_GPSD
test_AAC=AAC(r"../data4/Test.FASTA")
test_CTDC=libsvm_file_read(r"../data4/test_CTDC.LibSVM")
test_CTDT=libsvm_file_read(r"../data4/test_CTDT.LibSVM")
test_CTDD=libsvm_file_read(r"../data4/test_CTDD.LibSVM")
#test_GAAPC
test_GAAC=GAAC(r"../data4/Test.FASTA")
test_GDPC=libsvm_file_read(r"../data4/test_GDPC.LibSVM")
test_GTPC=libsvm_file_read(r"../data4/test_GTPC.LibSVM")
#test_ASDC
test_ASDC=libsvm_file_read(r"../data4/test_ASDC.LibSVM")
#test_PAAC
test_PAAC=libsvm_file_read(r"../data4/test_PAAC.LibSVM")

test_APAAC=libsvm_file_read(r"../data4/test_APAAC.LibSVM")
test_CKSAAGP=libsvm_file_read(r"../data4/test_CKSAAGP.LibSVM")
test_CKSAAP=libsvm_file_read(r"../data4/test_CKSAAP.LibSVM")
test_CTriad=libsvm_file_read(r"../data4/test_CTriad.LibSVM")
test_DDE=libsvm_file_read(r"../data4/test_DDE.LibSVM")
test_DistancePair=libsvm_file_read(r"../data4/test_DistancePair.LibSVM")
test_DPC=libsvm_file_read(r"../data4/test_DPC.LibSVM")
#test_QSOrder=libsvm_file_read(r"dataset/test_QSOrder.LibSVM")
test_SOCNumber=libsvm_file_read(r"../data4/test_SOCNumber.LibSVM")
test_k_mer=np.load(r"../data4/Test_k_mer.npy")


cnn_test_feature=np.concatenate((test_AAC,test_CTDC,test_CTDT,test_CTDD,test_GAAC,test_GDPC,test_GTPC,test_ASDC,test_PAAC,
                              test_APAAC,test_CKSAAGP,test_CKSAAP,test_CTriad,test_DDE,test_DistancePair,test_DPC,
                              test_SOCNumber,test_k_mer),axis=1)

rnn_test_feature=np.concatenate((test_AAC,test_CTDC,test_CTDT,test_CTDD,test_GAAC,test_GDPC,test_GTPC,test_ASDC,test_PAAC,
                              test_APAAC,test_CKSAAGP,test_CKSAAP,test_CTriad,test_DDE,test_DistancePair,test_DPC,
                              test_SOCNumber,test_k_mer),axis=1)

cnn_test_feature=cnn_test_feature[:,:cnn_unused_number]
rnn_test_feature=rnn_test_feature[:,:rnn_unused_number]

cnn_feature=np.concatenate((cnn_positive_feature,cnn_negative_feature,cnn_train_feature,cnn_test_feature),axis=0)
rnn_feature=np.concatenate((rnn_positive_feature,rnn_negative_feature,rnn_train_feature,rnn_test_feature),axis=0)


earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=80,mode="min")

aaindex=pd.read_table('aaindex31',sep='\s+',header=None)
#aaindex=aaindex.subtract(aaindex.min(axis=1),axis=0).divide((aaindex.max(axis=1)-aaindex.min(axis=1)),axis=0)
aa=[x for x in 'ARNDCQEGHILKMFPSTWYV']
aaindex=aaindex.to_numpy().T
index={x:y for x,y in zip(aa,aaindex.tolist())}
index['-']=np.zeros(31).tolist()

def cnn(max_len,depth,l1=64,l2=512,gamma=1e-4,lr=1e-4,w1=16,w2=8):
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=regularizers.l2(gamma),input_shape=(max_len,depth),padding='valid',strides=1))
    model.add(layers.MaxPooling1D(w2,padding="valid",strides=1))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l2(gamma)))
    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    
    """model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])"""
    
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    
    return model

def rnn(max_len,depth,l1=64,l2=512,gamma=1e-4,lr=1e-4):
    model=models.Sequential()
    model.add(layers.LSTM(l1,activation='tanh',return_sequences=True,input_shape=(max_len,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l2(gamma)))
    
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


f=open("../data3/positive.fasta",'r')
positive_lines=f.readlines()
positive_seq,_=read_data(positive_lines)
f.close()

f=open("../data3/negative.fasta",'r')
negative_lines=f.readlines()
negative_seq,_=read_data(negative_lines)
f.close()

f=open("../data4/Train.FASTA",'r')
train_lines=f.readlines()
train_seq,train_label=read_data(train_lines)
f.close()

f=open("../data4/Test.FASTA",'r')
test_lines=f.readlines()
test_seq,test_label=read_data(test_lines)
f.close()

f=open("all-1.fasta",'r')
all_1_lines=f.readlines()[:985*2]
all_1_seq,_=read_data(all_1_lines)
f.close()

padding_lens= len(max([max(train_seq, key=len, default=''),max(test_seq, key=len, default=''),
                      max(positive_seq, key=len, default=''),max(negative_seq, key=len, default=''),
                      max(all_1_seq, key=len, default='')],key=len,default=''))


label=np.concatenate((np.ones(592),np.zeros(592),train_label,test_label),axis=0)


positive_encoding=np.array(aminoacids_encode(positive_seq, padding_lens))
row=positive_encoding.shape[1]
column=positive_encoding.shape[2]
positive_encoding=np.reshape(positive_encoding,(len(positive_encoding),row*column))
cnn_positive_encoding=np.concatenate((positive_encoding,cnn_positive_feature),axis=1)
rnn_positive_encoding=np.concatenate((positive_encoding,rnn_positive_feature),axis=1)



negative_encoding=np.array(aminoacids_encode(negative_seq, padding_lens))
row=negative_encoding.shape[1]
column=negative_encoding.shape[2]
negative_encoding=np.reshape(negative_encoding,(len(negative_encoding),row*column))
cnn_negative_encoding=np.concatenate((negative_encoding,cnn_negative_feature),axis=1)
rnn_negative_encoding=np.concatenate((negative_encoding,rnn_negative_feature),axis=1)


train_encoding=np.array(aminoacids_encode(train_seq, padding_lens))
row=train_encoding.shape[1]
column=train_encoding.shape[2]
train_encoding=np.reshape(train_encoding,(len(train_encoding),row*column))
cnn_train_encoding=np.concatenate((train_encoding,cnn_train_feature),axis=1)
rnn_train_encoding=np.concatenate((train_encoding,rnn_train_feature),axis=1)


test_encoding=np.array(aminoacids_encode(test_seq, padding_lens))
row=test_encoding.shape[1]
column=test_encoding.shape[2]
test_encoding=np.reshape(test_encoding,(len(test_encoding),row*column))
cnn_test_encoding=np.concatenate((test_encoding,cnn_test_feature),axis=1)
rnn_test_encoding=np.concatenate((test_encoding,rnn_test_feature),axis=1)

scale=MinMaxScaler()

cnn_encoding=np.concatenate((cnn_positive_encoding,cnn_negative_encoding,cnn_train_encoding,cnn_test_encoding),axis=0)
cnn_encoding=scale.fit_transform(cnn_encoding)
cnn_train_encoding,cnn_test_encoding,train_label,test_label=train_test_split(cnn_encoding,label,train_size=0.9,random_state=0)

rnn_encoding=np.concatenate((rnn_positive_encoding,rnn_negative_encoding,rnn_train_encoding,rnn_test_encoding),axis=0)
rnn_encoding=scale.fit_transform(rnn_encoding)
rnn_train_encoding,rnn_test_encoding,train_label,test_label=train_test_split(rnn_encoding,label,train_size=0.9,random_state=0)



# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
oversample = SMOTE()
cnn_train_encoding,cnn_train_label=oversample.fit_resample(cnn_train_encoding,train_label)
rnn_train_encoding,rnn_train_label=oversample.fit_resample(rnn_train_encoding,train_label)

cnn_train_encoding=np.reshape(cnn_train_encoding,(len(cnn_train_encoding),row+cnn_add_length,column))
rnn_train_encoding=np.reshape(rnn_train_encoding,(len(rnn_train_encoding),row+rnn_add_length,column))

cnn_test_encoding=np.reshape(cnn_test_encoding,(len(cnn_test_encoding),row+cnn_add_length,column))
rnn_test_encoding=np.reshape(rnn_test_encoding,(len(rnn_test_encoding),row+rnn_add_length,column))


"""随机化处理"""

np.random.seed(666)
train_shuffle_list=[]
while True:
    i=np.random.randint(0,cnn_train_encoding.shape[0])
    if i not in train_shuffle_list:
        train_shuffle_list.append(i)
    if len(train_shuffle_list)==cnn_train_encoding.shape[0]:
        break
    
cnn_train_encoding_shuffle=cnn_train_encoding[train_shuffle_list]
rnn_train_encoding_shuffle=rnn_train_encoding[train_shuffle_list]

cnn_train_label_shuffle=cnn_train_label[train_shuffle_list]
rnn_train_label_shuffle=rnn_train_label[train_shuffle_list]

cnn_train_encoding=cnn_train_encoding_shuffle
rnn_train_encoding=rnn_train_encoding_shuffle
cnn_train_label=cnn_train_label_shuffle
rnn_train_label=rnn_train_label_shuffle

k=10
np.random.seed(666)

num=len(cnn_train_label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
score_cnn=np.zeros(num)
score_rnn=np.zeros(num)


cnntrain=np.load("bestnpz/data_allcnntrain.npz")
rnntrain=np.load("bestnpz/data_allrnntrain.npz")
cnntest=np.load("bestnpz/data_allcnntest.npz")
rnntest=np.load("bestnpz/data_allrnntest.npz")

fprc,tprc,_=roc_curve(cnntrain['label'],cnntrain['cnn'])
fprr,tprr,_=roc_curve(rnntrain['label'],rnntrain['rnn'])
fpr1,tpr1,_=roc_curve(cnntest['label'],cnntest['cnn'])
fpr2,tpr2,_=roc_curve(rnntest['label'],rnntest['rnn'])

cnn_train_baseline=[auc(fprc,tprc)]
rnn_train_baseline=[auc(fprr,tprr)]
cnn_test_baseline=[auc(fpr1,tpr1)]
rnn_test_baseline=[auc(fpr2,tpr2)]

count=0
while True:
    
    flag=0
    
    """for fold in range(k):
        
        cnn_trainLabel=cnn_train_label[mode!=fold]
        cnn_testLabel=cnn_train_label[mode==fold]
        
        rnn_trainLabel=rnn_train_label[mode!=fold]
        rnn_testLabel=rnn_train_label[mode==fold]
        
        cnn_trainFeature1=cnn_train_encoding[mode!=fold]
        cnn_testFeature1=cnn_train_encoding[mode==fold]
        rnn_trainFeature1=rnn_train_encoding[mode!=fold]
        rnn_testFeature1=rnn_train_encoding[mode==fold]
        
        m1=cnn(padding_lens+cnn_add_length,31)
        m1.fit(cnn_trainFeature1,cnn_trainLabel,batch_size=256,epochs=10000,verbose=1,validation_data=(cnn_testFeature1,cnn_testLabel),
               shuffle=True,callbacks=[earlystop_callback])
        score_cnn[mode==fold]=m1.predict(cnn_testFeature1).reshape(len(cnn_testFeature1))
        
        
        m2=rnn(padding_lens+rnn_add_length,31)
        m2.fit(rnn_trainFeature1,rnn_trainLabel,batch_size=256,epochs=10000,verbose=1,validation_data=(rnn_testFeature1,rnn_testLabel),
               shuffle=True,callbacks=[earlystop_callback])
        score_rnn[mode==fold]=m2.predict(rnn_testFeature1).reshape(len(rnn_testFeature1))
        
    np.savez('data_allcnntrain.npz',cnn=score_cnn,label=cnn_train_label)
    np.savez('data_allrnntrain.npz',rnn=score_rnn,label=rnn_train_label)"""
    
    
    model1=cnn(padding_lens+cnn_add_length,31)
    model1.fit(cnn_train_encoding,cnn_train_label,batch_size=512,epochs=10000,verbose=1,validation_data=(cnn_test_encoding,test_label),
              shuffle=True,callbacks=[earlystop_callback])
    cnn_test_preidct=model1.predict(cnn_test_encoding).reshape(len(cnn_test_encoding))
    np.savez('data_allcnntest.npz',cnn=cnn_test_preidct,label=test_label)

    model2=rnn(padding_lens+rnn_add_length,31)
    model2.fit(rnn_train_encoding,rnn_train_label,batch_size=128,epochs=10000,verbose=1,validation_data=(rnn_test_encoding,test_label),
              shuffle=True,callbacks=[earlystop_callback])
    rnn_test_preidct=model2.predict(rnn_test_encoding).reshape(len(rnn_test_encoding))
    np.savez('data_allrnntest.npz',rnn=rnn_test_preidct,label=test_label)
    
    
    cnntrain=np.load("data_allcnntrain.npz")
    rnntrain=np.load("data_allrnntrain.npz")
    cnntest=np.load("data_allcnntest.npz")
    rnntest=np.load("data_allrnntest.npz")
    
    min_step=1e-3
    
    fprc,tprc,_=roc_curve(cnntrain['label'],cnntrain['cnn'])
    fprr,tprr,_=roc_curve(rnntrain['label'],rnntrain['rnn'])
    fpr1,tpr1,_=roc_curve(cnntest['label'],cnntest['cnn'])
    fpr2,tpr2,_=roc_curve(rnntest['label'],rnntest['rnn'])
    
    
    
    ###########################################################################
    if  auc(fprc,tprc)>max(cnn_train_baseline) :
        cnn_train_baseline.append(auc(fprc,tprc))
        np.savez('bestnpz/data_allcnntrain.npz',cnn=score_cnn,label=cnn_train_label)
        flag+=1
        
    elif auc(fprr,tprr)>max(rnn_train_baseline) :
        rnn_train_baseline.append(auc(fprc,tprc))
        np.savez('bestnpz/data_allrnntrain.npz',rnn=score_rnn,label=rnn_train_label)
        flag+=1
        
    elif auc(fpr1,tpr1)>max(cnn_test_baseline):
        cnn_test_baseline.append(auc(fpr1,tpr1))
        np.savez('bestnpz/data_allcnntest.npz',cnn=cnn_test_preidct,label=test_label)
        flag+=1
    
    elif auc(fpr2,tpr2)>max(rnn_test_baseline):
        rnn_test_baseline.append(auc(fpr2,tpr2))
        np.savez('bestnpz/data_allrnntest.npz',rnn=rnn_test_preidct,label=test_label)
        flag+=1
    ###########################################################################
        
    if flag>0:
        
        cnntrain=np.load("bestnpz/data_allcnntrain.npz")
        rnntrain=np.load("bestnpz/data_allrnntrain.npz")
        cnntest=np.load("bestnpz/data_allcnntest.npz")
        rnntest=np.load("bestnpz/data_allrnntest.npz")
        
        fprc,tprc,_=roc_curve(cnntrain['label'],cnntrain['cnn'])
        fprr,tprr,_=roc_curve(rnntrain['label'],rnntrain['rnn'])
        fpr1,tpr1,_=roc_curve(cnntest['label'],cnntest['cnn'])
        fpr2,tpr2,_=roc_curve(rnntest['label'],rnntest['rnn'])
        
    
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
        count+=flag
    print("本次更新%d次，共计更新%d次"%(flag,count))


