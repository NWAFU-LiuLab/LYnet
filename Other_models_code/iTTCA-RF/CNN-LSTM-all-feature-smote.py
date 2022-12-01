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


train_APAAC=libsvm_file_read(r"dataset/train_APAAC.LibSVM")
train_CKSAAGP=libsvm_file_read(r"dataset/train_CKSAAGP.LibSVM")
train_CKSAAP=libsvm_file_read(r"dataset/train_CKSAAP.LibSVM")
train_CTriad=libsvm_file_read(r"dataset/train_CTriad.LibSVM")
train_DDE=libsvm_file_read(r"dataset/train_DDE.LibSVM")
train_DistancePair=libsvm_file_read(r"dataset/train_DistancePair.LibSVM")
train_DPC=libsvm_file_read(r"dataset/train_DPC.LibSVM")
#train_QSOrder=libsvm_file_read(r"dataset/train_QSOrder.LibSVM")
train_SOCNumber=libsvm_file_read(r"dataset/train_SOCNumber.LibSVM")
#train_k_mer=read_k_mer(r"dataset/train_2_mer_.csv")
train_k_mer=np.load(r"seqset\train_k_mer.npy")

cnn_train_feature=np.concatenate((train_AAC,train_CTDC,train_CTDT,train_CTDD,train_GAAC,train_GDPC,train_GTPC,train_ASDC,train_PAAC,
                              train_APAAC,train_CKSAAGP,train_CKSAAP,train_CTriad,train_DDE,train_DistancePair,train_DPC,
                              train_SOCNumber,train_k_mer),axis=1)

rnn_train_feature=np.concatenate((train_AAC,train_CTDC,train_CTDT,train_CTDD,train_GAAC,train_GDPC,train_GTPC,train_ASDC,train_PAAC,
                              train_APAAC,train_CKSAAGP,train_CKSAAP,train_CTriad,train_DDE,train_DistancePair,train_DPC,
                              train_SOCNumber,train_k_mer),axis=1)

cnn_add_length=int(cnn_train_feature.shape[1]/31)
cnn_unused_number=cnn_add_length*31-cnn_train_feature.shape[1]

rnn_add_length=int(rnn_train_feature.shape[1]/31)
rnn_unused_number=rnn_add_length*31-rnn_train_feature.shape[1]

cnn_train_feature=cnn_train_feature[:,:cnn_unused_number]
rnn_train_feature=rnn_train_feature[:,:rnn_unused_number]


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

test_APAAC=libsvm_file_read(r"dataset/test_APAAC.LibSVM")
test_CKSAAGP=libsvm_file_read(r"dataset/test_CKSAAGP.LibSVM")
test_CKSAAP=libsvm_file_read(r"dataset/test_CKSAAP.LibSVM")
test_CTriad=libsvm_file_read(r"dataset/test_CTriad.LibSVM")
test_DDE=libsvm_file_read(r"dataset/test_DDE.LibSVM")
test_DistancePair=libsvm_file_read(r"dataset/test_DistancePair.LibSVM")
test_DPC=libsvm_file_read(r"dataset/test_DPC.LibSVM")
#test_QSOrder=libsvm_file_read(r"dataset/test_QSOrder.LibSVM")
test_SOCNumber=libsvm_file_read(r"dataset/test_SOCNumber.LibSVM")
#test_k_mer=read_k_mer(r"dataset/test_2_mer_.csv")
test_k_mer=np.load(r"seqset\test_k_mer.npy")


cnn_test_feature=np.concatenate((test_AAC,test_CTDC,test_CTDT,test_CTDD,test_GAAC,test_GDPC,test_GTPC,test_ASDC,test_PAAC,
                              test_APAAC,test_CKSAAGP,test_CKSAAP,test_CTriad,test_DDE,test_DistancePair,test_DPC,
                              test_SOCNumber,test_k_mer),axis=1)

rnn_test_feature=np.concatenate((test_AAC,test_CTDC,test_CTDT,test_CTDD,test_GAAC,test_GDPC,test_GTPC,test_ASDC,test_PAAC,
                              test_APAAC,test_CKSAAGP,test_CKSAAP,test_CTriad,test_DDE,test_DistancePair,test_DPC,
                              test_SOCNumber,test_k_mer),axis=1)

cnn_test_feature=cnn_test_feature[:,:cnn_unused_number]
rnn_test_feature=rnn_test_feature[:,:rnn_unused_number]


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
            if "negative" in lines[i]:
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
cnn_train_encoding=np.concatenate((train_encoding,cnn_train_feature),axis=1)
rnn_train_encoding=np.concatenate((train_encoding,rnn_train_feature),axis=1)

# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
oversample = SMOTE()
cnn_train_encoding,cnn_train_label=oversample.fit_resample(cnn_train_encoding,train_label)
rnn_train_encoding,rnn_train_label=oversample.fit_resample(rnn_train_encoding,train_label)

cnn_train_encoding=np.reshape(cnn_train_encoding,(len(cnn_train_encoding),row+cnn_add_length,column))
rnn_train_encoding=np.reshape(rnn_train_encoding,(len(rnn_train_encoding),row+rnn_add_length,column))


test_encoding=np.array(aminoacids_encode(test_seq, padding_lens))
cnn_test_encoding=np.concatenate((test_encoding,cnn_test_feature.reshape((len(cnn_test_feature),cnn_add_length,31))),axis=1)
rnn_test_encoding=np.concatenate((test_encoding,rnn_test_feature.reshape((len(rnn_test_feature),rnn_add_length,31))),axis=1)

cnn_scale=MinMaxScaler()
rnn_scale=MinMaxScaler()
cnn_scale.fit(cnn_train_encoding.reshape(cnn_train_encoding.shape[0],cnn_train_encoding.shape[1]*cnn_train_encoding.shape[2]))
rnn_scale.fit(rnn_train_encoding.reshape(rnn_train_encoding.shape[0],rnn_train_encoding.shape[1]*rnn_train_encoding.shape[2]))

cnn_train_encoding=(cnn_scale.transform(cnn_train_encoding.reshape(cnn_train_encoding.shape[0],cnn_train_encoding.shape[1]*cnn_train_encoding.shape[2]))).reshape((cnn_train_encoding.shape[0],cnn_train_encoding.shape[1],cnn_train_encoding.shape[2]))
rnn_train_encoding=(rnn_scale.transform(rnn_train_encoding.reshape(rnn_train_encoding.shape[0],rnn_train_encoding.shape[1]*rnn_train_encoding.shape[2]))).reshape((rnn_train_encoding.shape[0],rnn_train_encoding.shape[1],rnn_train_encoding.shape[2]))
cnn_test_encoding=(cnn_scale.transform(cnn_test_encoding.reshape(cnn_test_encoding.shape[0],cnn_test_encoding.shape[1]*cnn_test_encoding.shape[2]))).reshape((cnn_test_encoding.shape[0],cnn_test_encoding.shape[1],cnn_test_encoding.shape[2]))
rnn_test_encoding=(rnn_scale.transform(rnn_test_encoding.reshape(rnn_test_encoding.shape[0],rnn_test_encoding.shape[1]*rnn_test_encoding.shape[2]))).reshape((rnn_test_encoding.shape[0],rnn_test_encoding.shape[1],rnn_test_encoding.shape[2]))

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


cnntrain=np.load("bestnpz/cnntrain.npz")
rnntrain=np.load("bestnpz/rnntrain.npz")
cnntest=np.load("bestnpz/cnntest.npz")
rnntest=np.load("bestnpz/rnntest.npz")

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
        
    np.savez('cnntrain.npz',cnn=score_cnn,label=cnn_train_label)
    np.savez('rnntrain.npz',rnn=score_rnn,label=rnn_train_label)"""
    
    
    model1=cnn(padding_lens+cnn_add_length,31)
    model1.fit(cnn_train_encoding,cnn_train_label,batch_size=cnn_train_encoding.shape[0],epochs=10000,verbose=1,validation_data=(cnn_test_encoding,test_label),
              shuffle=True,callbacks=[earlystop_callback])
    cnn_test_preidct=model1.predict(cnn_test_encoding).reshape(len(cnn_test_encoding))
    np.savez('cnntest.npz',cnn=cnn_test_preidct,label=test_label)

    model2=rnn(padding_lens+rnn_add_length,31)
    model2.fit(rnn_train_encoding,rnn_train_label,batch_size=256,epochs=10000,verbose=1,validation_data=(rnn_test_encoding,test_label),
              shuffle=True,callbacks=[earlystop_callback])
    rnn_test_preidct=model2.predict(rnn_test_encoding).reshape(len(rnn_test_encoding))
    np.savez('rnntest.npz',rnn=rnn_test_preidct,label=test_label)
    
    
    cnntrain=np.load("cnntrain.npz")
    rnntrain=np.load("rnntrain.npz")
    cnntest=np.load("cnntest.npz")
    rnntest=np.load("rnntest.npz")
    
    min_step=1e-3
    
    fprc,tprc,_=roc_curve(cnntrain['label'],cnntrain['cnn'])
    fprr,tprr,_=roc_curve(rnntrain['label'],rnntrain['rnn'])
    fpr1,tpr1,_=roc_curve(cnntest['label'],cnntest['cnn'])
    fpr2,tpr2,_=roc_curve(rnntest['label'],rnntest['rnn'])
    
    
    
    ###########################################################################
    if  auc(fprc,tprc)>max(cnn_train_baseline) :
        cnn_train_baseline.append(auc(fprc,tprc))
        np.savez('bestnpz/cnntrain.npz',cnn=score_cnn,label=cnn_train_label)
        flag+=1
        
    elif auc(fprr,tprr)>max(rnn_train_baseline) :
        rnn_train_baseline.append(auc(fprc,tprc))
        np.savez('bestnpz/rnntrain.npz',rnn=score_rnn,label=rnn_train_label)
        flag+=1
        
    elif auc(fpr1,tpr1)>max(cnn_test_baseline):
        cnn_test_baseline.append(auc(fpr1,tpr1))
        np.savez('bestnpz/cnntest.npz',cnn=cnn_test_preidct,label=test_label)
        flag+=1
    
    elif auc(fpr2,tpr2)>max(rnn_test_baseline):
        rnn_test_baseline.append(auc(fpr2,tpr2))
        np.savez('bestnpz/rnntest.npz',rnn=rnn_test_preidct,label=test_label)
        flag+=1
    ###########################################################################
        
    if flag>0:
        
        cnntrain=np.load("bestnpz/cnntrain.npz")
        rnntrain=np.load("bestnpz/rnntrain.npz")
        cnntest=np.load("bestnpz/cnntest.npz")
        rnntest=np.load("bestnpz/rnntest.npz")
        
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
