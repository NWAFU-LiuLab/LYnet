# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 12:24:46 2022

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

#train_positive_GPSD
train_positive_AAC=AAC(r"data2/Train_Positive.FASTA")
train_positive_CTDC=libsvm_file_read(r"data2/Train_Positive_CTDC.LibSVM")
train_positive_CTDT=libsvm_file_read(r"data2/Train_Positive_CTDT.LibSVM")
train_positive_CTDD=libsvm_file_read(r"data2/Train_Positive_CTDD.LibSVM")
#train_positive_GAAPC
train_positive_GAAC=GAAC(r"data2/Train_Positive.FASTA")
train_positive_GDPC=libsvm_file_read(r"data2/Train_Positive_GDPC.LibSVM")
train_positive_GTPC=libsvm_file_read(r"data2/Train_Positive_GTPC.LibSVM")
#train_positive_ASDC
train_positive_ASDC=libsvm_file_read(r"data2/Train_Positive_ASDC.LibSVM")
#train_positive_PAAC
train_positive_PAAC=libsvm_file_read(r"data2/Train_Positive_PAAC.LibSVM")


train_positive_APAAC=libsvm_file_read(r"data2/Train_Positive_APAAC.LibSVM")
train_positive_CKSAAGP=libsvm_file_read(r"data2/Train_Positive_CKSAAGP.LibSVM")
train_positive_CKSAAP=libsvm_file_read(r"data2/Train_Positive_CKSAAP.LibSVM")
train_positive_CTriad=libsvm_file_read(r"data2/Train_Positive_CTriad.LibSVM")
train_positive_DDE=libsvm_file_read(r"data2/Train_Positive_DDE.LibSVM")
train_positive_DistancePair=libsvm_file_read(r"data2/Train_Positive_DistancePair.LibSVM")
train_positive_DPC=libsvm_file_read(r"data2/Train_Positive_DPC.LibSVM")
#train_QSOrder=libsvm_file_read(r"dataset/train_QSOrder.LibSVM")
train_positive_SOCNumber=libsvm_file_read(r"data2/Train_Positive_SOCNumber.LibSVM")
train_positive_k_mer=read_k_mer(r"data2/Train_Positive_2_mer_.csv")

cnn_train_positive_feature=np.concatenate((train_positive_GDPC,train_positive_GTPC,train_positive_ASDC,
                              train_positive_APAAC,train_positive_CKSAAGP),axis=1)

rnn_train_positive_feature=np.concatenate((train_positive_GDPC,train_positive_GTPC,train_positive_ASDC,
                              train_positive_APAAC,train_positive_CKSAAGP),axis=1)

cnn_add_length=int(cnn_train_positive_feature.shape[1]/31)
cnn_unused_number=cnn_add_length*31-cnn_train_positive_feature.shape[1]

rnn_add_length=int(rnn_train_positive_feature.shape[1]/31)
rnn_unused_number=rnn_add_length*31-rnn_train_positive_feature.shape[1]

cnn_train_positive_feature=cnn_train_positive_feature[:,:cnn_unused_number]
rnn_train_positive_feature=rnn_train_positive_feature[:,:rnn_unused_number]
scale=MinMaxScaler()
cnn_train_positive_feature=scale.fit_transform(cnn_train_positive_feature)
rnn_train_positive_feature=scale.fit_transform(rnn_train_positive_feature)


#train_negative_GPSD
train_negative_AAC=AAC(r"data2/Train_Negative.FASTA")
train_negative_CTDC=libsvm_file_read(r"data2/Train_Negative_CTDC.LibSVM")
train_negative_CTDT=libsvm_file_read(r"data2/Train_Negative_CTDT.LibSVM")
train_negative_CTDD=libsvm_file_read(r"data2/Train_Negative_CTDD.LibSVM")
#train_negative_GAAPC
train_negative_GAAC=GAAC(r"data2/Train_Negative.FASTA")
train_negative_GDPC=libsvm_file_read(r"data2/Train_Negative_GDPC.LibSVM")
train_negative_GTPC=libsvm_file_read(r"data2/Train_Negative_GTPC.LibSVM")
#train_negative_ASDC
train_negative_ASDC=libsvm_file_read(r"data2/Train_Negative_ASDC.LibSVM")
#train_negative_PAAC
train_negative_PAAC=libsvm_file_read(r"data2/Train_Negative_PAAC.LibSVM")


train_negative_APAAC=libsvm_file_read(r"data2/Train_Negative_APAAC.LibSVM")
train_negative_CKSAAGP=libsvm_file_read(r"data2/Train_Negative_CKSAAGP.LibSVM")
train_negative_CKSAAP=libsvm_file_read(r"data2/Train_Negative_CKSAAP.LibSVM")
train_negative_CTriad=libsvm_file_read(r"data2/Train_Negative_CTriad.LibSVM")
train_negative_DDE=libsvm_file_read(r"data2/Train_Negative_DDE.LibSVM")
train_negative_DistancePair=libsvm_file_read(r"data2/Train_Negative_DistancePair.LibSVM")
train_negative_DPC=libsvm_file_read(r"data2/Train_Negative_DPC.LibSVM")
#train_QSOrder=libsvm_file_read(r"dataset/train_QSOrder.LibSVM")
train_negative_SOCNumber=libsvm_file_read(r"data2/Train_Negative_SOCNumber.LibSVM")
train_negative_k_mer=read_k_mer(r"data2/Train_Negative_2_mer_.csv")

cnn_train_negative_feature=np.concatenate((train_negative_GDPC,train_negative_GTPC,train_negative_ASDC,
                              train_negative_APAAC,train_negative_CKSAAGP),axis=1)

rnn_train_negative_feature=np.concatenate((train_negative_GDPC,train_negative_GTPC,train_negative_ASDC,
                              train_negative_APAAC,train_negative_CKSAAGP),axis=1)

cnn_add_length=int(cnn_train_negative_feature.shape[1]/31)
cnn_unused_number=cnn_add_length*31-cnn_train_negative_feature.shape[1]

rnn_add_length=int(rnn_train_negative_feature.shape[1]/31)
rnn_unused_number=rnn_add_length*31-rnn_train_negative_feature.shape[1]



cnn_train_negative_feature=cnn_train_negative_feature[:,:cnn_unused_number]
rnn_train_negative_feature=rnn_train_negative_feature[:,:rnn_unused_number]
scale=MinMaxScaler()
cnn_train_negative_feature=scale.fit_transform(cnn_train_negative_feature)
rnn_train_negative_feature=scale.fit_transform(rnn_train_negative_feature)

#test_positive_GPSD
test_positive_AAC=AAC(r"data2/Test_Positive.FASTA")
test_positive_CTDC=libsvm_file_read(r"data2/Test_Positive_CTDC.LibSVM")
test_positive_CTDT=libsvm_file_read(r"data2/Test_Positive_CTDT.LibSVM")
test_positive_CTDD=libsvm_file_read(r"data2/Test_Positive_CTDD.LibSVM")
#test_positive_GAAPC
test_positive_GAAC=GAAC(r"data2/Test_Positive.FASTA")
test_positive_GDPC=libsvm_file_read(r"data2/Test_Positive_GDPC.LibSVM")
test_positive_GTPC=libsvm_file_read(r"data2/Test_Positive_GTPC.LibSVM")
#test_positive_ASDC
test_positive_ASDC=libsvm_file_read(r"data2/Test_Positive_ASDC.LibSVM")
#test_positive_PAAC
test_positive_PAAC=libsvm_file_read(r"data2/Test_Positive_PAAC.LibSVM")


test_positive_APAAC=libsvm_file_read(r"data2/Test_Positive_APAAC.LibSVM")
test_positive_CKSAAGP=libsvm_file_read(r"data2/Test_Positive_CKSAAGP.LibSVM")
test_positive_CKSAAP=libsvm_file_read(r"data2/Test_Positive_CKSAAP.LibSVM")
test_positive_CTriad=libsvm_file_read(r"data2/Test_Positive_CTriad.LibSVM")
test_positive_DDE=libsvm_file_read(r"data2/Test_Positive_DDE.LibSVM")
test_positive_DistancePair=libsvm_file_read(r"data2/Test_Positive_DistancePair.LibSVM")
test_positive_DPC=libsvm_file_read(r"data2/Test_Positive_DPC.LibSVM")
#train_QSOrder=libsvm_file_read(r"dataset/train_QSOrder.LibSVM")
test_positive_SOCNumber=libsvm_file_read(r"data2/Test_Positive_SOCNumber.LibSVM")
test_positive_k_mer=read_k_mer(r"data2/Test_Positive_2_mer_.csv")

cnn_test_positive_feature=np.concatenate((test_positive_GDPC,test_positive_GTPC,test_positive_ASDC,
                              test_positive_APAAC,test_positive_CKSAAGP),axis=1)

rnn_test_positive_feature=np.concatenate((test_positive_GDPC,test_positive_GTPC,test_positive_ASDC,
                              test_positive_APAAC,test_positive_CKSAAGP),axis=1)

cnn_add_length=int(cnn_test_positive_feature.shape[1]/31)
cnn_unused_number=cnn_add_length*31-cnn_test_positive_feature.shape[1]

rnn_add_length=int(rnn_test_positive_feature.shape[1]/31)
rnn_unused_number=rnn_add_length*31-rnn_test_positive_feature.shape[1]

cnn_test_positive_feature=cnn_test_positive_feature[:,:cnn_unused_number]
rnn_test_positive_feature=rnn_test_positive_feature[:,:rnn_unused_number]
scale=MinMaxScaler()
cnn_test_positive_feature=scale.fit_transform(cnn_test_positive_feature)
rnn_test_positive_feature=scale.fit_transform(rnn_test_positive_feature)


#test_negative_GPSD
test_negative_AAC=AAC(r"data2/Test_Negative.FASTA")
test_negative_CTDC=libsvm_file_read(r"data2/Test_Negative_CTDC.LibSVM")
test_negative_CTDT=libsvm_file_read(r"data2/Test_Negative_CTDT.LibSVM")
test_negative_CTDD=libsvm_file_read(r"data2/Test_Negative_CTDD.LibSVM")
#test_negative_GAAPC
test_negative_GAAC=GAAC(r"data2/Test_Negative.FASTA")
test_negative_GDPC=libsvm_file_read(r"data2/Test_Negative_GDPC.LibSVM")
test_negative_GTPC=libsvm_file_read(r"data2/Test_Negative_GTPC.LibSVM")
#test_negative_ASDC
test_negative_ASDC=libsvm_file_read(r"data2/Test_Negative_ASDC.LibSVM")
#test_negative_PAAC
test_negative_PAAC=libsvm_file_read(r"data2/Test_Negative_PAAC.LibSVM")


test_negative_APAAC=libsvm_file_read(r"data2/Test_Negative_APAAC.LibSVM")
test_negative_CKSAAGP=libsvm_file_read(r"data2/Test_Negative_CKSAAGP.LibSVM")
test_negative_CKSAAP=libsvm_file_read(r"data2/Test_Negative_CKSAAP.LibSVM")
test_negative_CTriad=libsvm_file_read(r"data2/Test_Negative_CTriad.LibSVM")
test_negative_DDE=libsvm_file_read(r"data2/Test_Negative_DDE.LibSVM")
test_negative_DistancePair=libsvm_file_read(r"data2/Test_Negative_DistancePair.LibSVM")
test_negative_DPC=libsvm_file_read(r"data2/Test_Negative_DPC.LibSVM")
#train_QSOrder=libsvm_file_read(r"dataset/train_QSOrder.LibSVM")
test_negative_SOCNumber=libsvm_file_read(r"data2/Test_Negative_SOCNumber.LibSVM")
test_negative_k_mer=read_k_mer(r"data2/Test_Negative_2_mer_.csv")

cnn_test_negative_feature=np.concatenate((test_negative_GDPC,test_negative_GTPC,test_negative_ASDC,
                              test_negative_APAAC,test_negative_CKSAAGP),axis=1)

rnn_test_negative_feature=np.concatenate((test_negative_GDPC,test_negative_GTPC,test_negative_ASDC,
                              test_negative_APAAC,test_negative_CKSAAGP),axis=1)

cnn_add_length=int(cnn_test_negative_feature.shape[1]/31)
cnn_unused_number=cnn_add_length*31-cnn_test_negative_feature.shape[1]

rnn_add_length=int(rnn_test_negative_feature.shape[1]/31)
rnn_unused_number=rnn_add_length*31-rnn_test_negative_feature.shape[1]



cnn_test_negative_feature=cnn_test_negative_feature[:,:cnn_unused_number]
rnn_test_negative_feature=rnn_test_negative_feature[:,:rnn_unused_number]
scale=MinMaxScaler()
cnn_test_negative_feature=scale.fit_transform(cnn_test_negative_feature)
rnn_test_negative_feature=scale.fit_transform(rnn_test_negative_feature)


earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=20,mode="min")

aaindex=pd.read_table('aaindex31',sep='\s+',header=None)
aaindex=aaindex.subtract(aaindex.min(axis=1),axis=0).divide((aaindex.max(axis=1)-aaindex.min(axis=1)),axis=0)
aa=[x for x in 'ARNDCQEGHILKMFPSTWYV']
aaindex=aaindex.to_numpy().T
index={x:y for x,y in zip(aa,aaindex.tolist())}
index['-']=np.zeros(31).tolist()

def cnn(max_len,depth,l1=64,l2=512,gamma=1e-4,lr=1e-4,w1=16,w2=8):
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth),padding='same'))
    model.add(layers.MaxPooling1D(w2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    
    """model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])"""
    
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    
    return model

def rnn(max_len,depth,l1=16,l2=256,l3=256,gamma=1e-4,lr=1e-4):
    model=models.Sequential()
    model.add(layers.LSTM(l1,activation='tanh',return_sequences=True,kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(l2,activation='tanh',kernel_regularizer=regularizers.l2(gamma)))
    #model.add(layers.Dense(l3,activation='tanh',kernel_regularizer=regularizers.l2(gamma)))
    
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

f=open("data2/Train_Positive.FASTA",'r')
train_positive_lines=f.readlines()
train_positive_seq,_=read_data(train_positive_lines)
f.close()

f=open("data2/Train_Negative.FASTA",'r')
train_negative_lines=f.readlines()
train_negative_seq,_=read_data(train_negative_lines)
f.close()

f=open("data2/Test_Positive.FASTA",'r')
test_positive_lines=f.readlines()
test_positive_seq,_=read_data(test_positive_lines)
f.close()

f=open("data2/Test_Negative.FASTA",'r')
test_negative_lines=f.readlines()
test_negative_seq,_=read_data(test_negative_lines)
f.close()

padding_lens= len(max([max(train_positive_seq, key=len, default=''),max(test_positive_seq, key=len, default=''),
                      max(test_negative_seq, key=len, default=''),max(train_negative_seq, key=len, default='')],key=len,default=''))

train_positive_encoding=np.array(aminoacids_encode(train_positive_seq, padding_lens))
row=train_positive_encoding.shape[1]
column=train_positive_encoding.shape[2]
train_positive_encoding=np.reshape(train_positive_encoding,(len(train_positive_encoding),row*column))
cnn_train_positive_encoding=np.concatenate((train_positive_encoding,cnn_train_positive_feature),axis=1)
rnn_train_positive_encoding=np.concatenate((train_positive_encoding,rnn_train_positive_feature),axis=1)

test_positive_encoding=np.array(aminoacids_encode(test_positive_seq, padding_lens))
row=test_positive_encoding.shape[1]
column=test_positive_encoding.shape[2]
test_positive_encoding=np.reshape(test_positive_encoding,(len(test_positive_encoding),row*column))
cnn_test_positive_encoding=np.concatenate((test_positive_encoding,cnn_test_positive_feature),axis=1)
rnn_test_positive_encoding=np.concatenate((test_positive_encoding,rnn_test_positive_feature),axis=1)

train_negative_encoding=np.array(aminoacids_encode(train_negative_seq, padding_lens))
row=train_negative_encoding.shape[1]
column=train_negative_encoding.shape[2]
train_negative_encoding=np.reshape(train_negative_encoding,(len(train_negative_encoding),row*column))
cnn_train_negative_encoding=np.concatenate((train_negative_encoding,cnn_train_negative_feature),axis=1)
rnn_train_negative_encoding=np.concatenate((train_negative_encoding,rnn_train_negative_feature),axis=1)

test_negative_encoding=np.array(aminoacids_encode(test_negative_seq, padding_lens))
row=test_negative_encoding.shape[1]
column=test_negative_encoding.shape[2]
test_negative_encoding=np.reshape(test_negative_encoding,(len(test_negative_encoding),row*column))
cnn_test_negative_encoding=np.concatenate((test_negative_encoding,cnn_test_negative_feature),axis=1)
rnn_test_negative_encoding=np.concatenate((test_negative_encoding,rnn_test_negative_feature),axis=1)

train_label=np.concatenate((np.ones(cnn_train_positive_encoding.shape[0]),np.zeros(cnn_train_negative_encoding.shape[0])),axis=0)
cnn_train_encoding=np.concatenate((cnn_train_positive_encoding,cnn_train_negative_encoding),axis=0)
rnn_train_encoding=np.concatenate((rnn_train_positive_encoding,rnn_train_negative_encoding),axis=0)

test_label=np.concatenate((np.ones(cnn_test_positive_encoding.shape[0]),np.zeros(cnn_test_negative_encoding.shape[0])),axis=0)
cnn_test_encoding=np.concatenate((cnn_test_positive_encoding,cnn_test_negative_encoding),axis=0)
rnn_test_encoding=np.concatenate((rnn_test_positive_encoding,rnn_test_negative_encoding),axis=0)


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
shuffle_list=[]
while True:
    i=np.random.randint(0,cnn_train_encoding.shape[0])
    if i not in shuffle_list:
        shuffle_list.append(i)
    if len(shuffle_list)==cnn_train_encoding.shape[0]:
        break

cnn_train_encoding_shuffle=cnn_train_encoding[shuffle_list]
rnn_train_encoding_shuffle=rnn_train_encoding[shuffle_list]

cnn_train_label_shuffle=cnn_train_label[shuffle_list]
rnn_train_label_shuffle=rnn_train_label[shuffle_list]

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
    
np.savez('data2cnntrain.npz',cnn=score_cnn,label=cnn_train_label)
np.savez('data2rnntrain.npz',rnn=score_rnn,label=rnn_train_label)


model1=cnn(padding_lens+cnn_add_length,31)
model1.fit(cnn_train_encoding,cnn_train_label,batch_size=256,epochs=10000,verbose=1,validation_data=(cnn_test_encoding,test_label),
          shuffle=True,callbacks=[earlystop_callback])
cnn_test_preidct=model1.predict(cnn_test_encoding).reshape(len(cnn_test_encoding))


np.savez('data2cnntest.npz',cnn=cnn_test_preidct,label=test_label)"""

model2=rnn(padding_lens+rnn_add_length,31)
model2.fit(rnn_train_encoding,rnn_train_label,batch_size=256,epochs=10000,verbose=1,validation_data=(rnn_test_encoding,test_label),
          shuffle=True,callbacks=[earlystop_callback])
rnn_test_preidct=model2.predict(rnn_test_encoding).reshape(len(rnn_test_encoding))
np.savez('data2rnntest.npz',rnn=rnn_test_preidct,label=test_label)


cnntrain=np.load("data2cnntrain.npz")
rnntrain=np.load("data2rnntrain.npz")
cnntest=np.load("data2cnntest.npz")
rnntest=np.load("data2rnntest.npz")

fprc,tprc,_=roc_curve(cnntrain['label'],cnntrain['cnn'])
fprr,tprr,_=roc_curve(rnntrain['label'],rnntrain['rnn'])
fprcr,tprcr,_=roc_curve(rnntrain['label'],(rnntrain['rnn']+cnntrain['cnn'])/2)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef    
train_predict=[]
for i in (rnntrain['rnn']+cnntrain['cnn'])/2:
    if i>0.5:
        train_predict.append(1)
    else:
        train_predict.append(0)
        
print("Ten fold validation acc:%.3lf"%accuracy_score(cnntrain['label'],train_predict))
print("Ten fold validation precision:%.3lf"%precision_score(cnntrain['label'],train_predict))
print("Ten fold validation recall:%.3lf"%recall_score(cnntrain['label'],train_predict))
print("Ten fold validation MCC:%.3lf"%matthews_corrcoef(cnntrain['label'],train_predict))




fpr1,tpr1,_=roc_curve(cnntest['label'],cnntest['cnn'])
fpr2,tpr2,_=roc_curve(rnntest['label'],rnntest['rnn'])
fpr3,tpr3,_=roc_curve(rnntest['label'],(rnntest['rnn']+cnntest['cnn'])/2)

test_predict=[]
for i in (rnntest['rnn']+cnntest['cnn'])/2:
    if i>0.5:
        test_predict.append(1)
    else:
        test_predict.append(0)
        
print("indepandent test acc：%.3lf"%accuracy_score(cnntest['label'],test_predict))
print("indepandent test precision：%.3lf"%precision_score(cnntest['label'],test_predict))
print("indepandent test recall：%.3lf"%recall_score(cnntest['label'],test_predict))
print("indepandent test MCC：%.3lf"%matthews_corrcoef(cnntest['label'],test_predict))

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
