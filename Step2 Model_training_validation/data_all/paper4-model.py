# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:17:15 2022

@author: lvyang
"""

from libsvm.svm import *
from libsvm.svmutil import *
import numpy as np
import re
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef
from sklearn.model_selection import train_test_split

def tanimoto_coefficient(p_vec, q_vec):
    """
    This method implements the cosine tanimoto coefficient metric
    :param p_vec: vector one
    :param q_vec: vector two
    :return: the tanimoto coefficient between vector one and two
    """
    pq = np.dot(p_vec, q_vec)
    p_square = np.linalg.norm(p_vec)
    q_square = np.linalg.norm(q_vec)
    return pq / (p_square + q_square - pq)

def MDMR(train_features,train_label,test_features):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.metrics import auc
    #step1:特征命名（如果有需要可能会用这个进行存储）
    feature_name=[str(i) for i in range(len(train_features))]
    
    #step2：MRMD得分对feature_array进行逆排序，得分越高的特征重要性越高，排的列数越靠前,
    RVi=[]
    DVi=[]
    MRMD_scorei=[]
    for i in range(0,train_features.shape[1]):#feature_array.shape[1]表示统计数组的列数
        DV=0.0
        RV=np.corrcoef(train_features[:,i], train_label)#计算x中每一列与y的皮尔逊相关系数
        RVi.append(RV[0,1])
        for j in range(0,train_features.shape[1]):
            EDij=np.linalg.norm(train_features[:,i]-train_features[:,j])#范数计算使用欧式距离公式
            CDij=np.dot(train_features[:,i],train_features[:,j])/(np.linalg.norm(train_features[:,i])*(np.linalg.norm(train_features[:,j])))#余弦距离
            TCij=tanimoto_coefficient(train_features[:,i], train_features[:,j])
            DV+=EDij
            DV+=CDij
            DV-=TCij
        DVi.append(DV/train_features.shape[1])#计算x中每一列到x中所有列的欧氏距离，然后再求平均值
    for k in range(train_features.shape[1]):
        MRMD_scorei.append(RVi[k]+ DVi[k])#得到MRMD得分
        
    """根据x中各个特征提取指标的MRMD得分，由大到小对各个指标进行排序
    即MRMD越大的指标,该列的数据越具有代表性，应优先让随机森林数据选择器使用"""
    
    MRMD_scorei = np.array(MRMD_scorei,dtype=np.float32)#转为数组
    rank=np.argsort(MRMD_scorei)#numpy.argsort() 函数返回的是数组值从小到大的索引值。
    rank=rank[::-1]  #rank逆序
    train_features_rank=train_features[...,rank] #例feature_array[...,1]表示取第一列元素
    
    #feature_array_copy=feature_array_copy[...,rank]
    
#    rank_list=list(rank)
    """对特征名字也进行排序与train_features_rank对应"""
#    feature_name_rank=[feature_name[x] for x in rank_list]
    
    #step3：用随机森林剔除冗余数据，寻找最高的AUC指标的特征组合
    MDMR_train_data,MDMR_test_data,MDMR_train_label,MDMR_test_label=train_test_split(train_features_rank,
                                                                                     train_label,test_size=0.2,
                                                                                     random_state=0,stratify=train_label)
    
    # 运行随机森林，寻找最高的AUC指标的特征组合，即 max_key的值
    conclusion = {}#创建检索字典
    
    for cnt in range(train_features_rank.shape[1]):
        
        X_train = MDMR_train_data[..., 0:cnt+1]#取前cnt项
        X_test = MDMR_test_data[..., 0:cnt+1]#取前cnt项
        
        clf = RandomForestClassifier(random_state=0,n_jobs=-1,verbose=0)
        clf = clf.fit(X_train,MDMR_train_label)
    
        prob_predict_y_test = clf.predict_proba(X_test)
        predictions_test = prob_predict_y_test[:, 1]
        
        fpr, tpr, thresholds = metrics.roc_curve(MDMR_test_label, predictions_test, pos_label=1)
        roc_auc = auc(fpr, tpr)
        conclusion[cnt] = roc_auc
        print('共%d次随机森林计算，已完成%s次' %(train_features_rank.shape[1],cnt))
    
    # 找到conclusion字典中最大值对应的键
    for key, value in conclusion.items():
        if value == max(conclusion.values()):
            max_key = key
    
    best_train_features=train_features_rank[:,0:max_key]
    best_test_features=test_features[...,rank][:,0:max_key]
    
    return best_train_features,best_test_features


def get_label(file_path):
    label=[]
    f=open(file_path,'r')
    lines=f.readlines()
    f.close()
    for i in range(len(lines)):
        if i%2==0:
            seq=[]
            seq.extend(lines[i][:-1])
            if "n" in seq:
                label.append(0)
            else:
                label.append(1)
    return np.array(label)
            
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

train_label=get_label(r"Train.FASTA")

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
positive_feature=np.concatenate((positive_AAC,positive_CTDC,positive_CTDT,positive_CTDD,positive_GAAC,positive_GDPC,positive_GTPC,positive_ASDC,positive_PAAC),axis=1)


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
negative_feature=np.concatenate((negative_AAC,negative_CTDC,negative_CTDT,negative_CTDD,negative_GAAC,negative_GDPC,negative_GTPC,negative_ASDC,negative_PAAC),axis=1)

#train_GPSD
train_AAC=AAC(r"Train.FASTA")
train_CTDC=libsvm_file_read(r"../data4/train_CTDC.LibSVM")
train_CTDT=libsvm_file_read(r"../data4/train_CTDT.LibSVM")
train_CTDD=libsvm_file_read(r"../data4/train_CTDD.LibSVM")
#train_GAAPC
train_GAAC=GAAC(r"Train.FASTA")
train_GDPC=libsvm_file_read(r"../data4/train_GDPC.LibSVM")
train_GTPC=libsvm_file_read(r"../data4/train_GTPC.LibSVM")
#train_ASDC
train_ASDC=libsvm_file_read(r"../data4/train_ASDC.LibSVM")
#train_PAAC
train_PAAC=libsvm_file_read(r"../data4/train_PAAC.LibSVM")

train_feature=np.concatenate((train_AAC,train_CTDC,train_CTDT,train_CTDD,train_GAAC,train_GDPC,train_GTPC,train_ASDC,train_PAAC),axis=1)

test_label=get_label(r"Test.FASTA")

#test_GPSD
test_AAC=AAC(r"Test.FASTA")
test_CTDC=libsvm_file_read(r"../data4/test_CTDC.LibSVM")
test_CTDT=libsvm_file_read(r"../data4/test_CTDT.LibSVM")
test_CTDD=libsvm_file_read(r"../data4/test_CTDD.LibSVM")
#test_GAAPC
test_GAAC=GAAC(r"Test.FASTA")
test_GDPC=libsvm_file_read(r"../data4/test_GDPC.LibSVM")
test_GTPC=libsvm_file_read(r"../data4/test_GTPC.LibSVM")
#test_ASDC
test_ASDC=libsvm_file_read(r"../data4/test_ASDC.LibSVM")
#test_PAAC
test_PAAC=libsvm_file_read(r"../data4/test_PAAC.LibSVM")

test_feature=np.concatenate((test_AAC,test_CTDC,test_CTDT,test_CTDD,test_GAAC,test_GDPC,test_GTPC,test_ASDC,test_PAAC),axis=1)

encoding=np.concatenate((positive_feature,negative_feature,train_feature,test_feature),axis=0)
label=np.concatenate((np.ones(592),np.zeros(592),train_label,test_label),axis=0)
train_feature,test_feature,train_label,test_label=train_test_split(encoding,label,train_size=0.8,random_state=22)

# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
oversample = SMOTE()
train_feature,train_label=oversample.fit_resample(train_feature,train_label)

#得到最佳特征
best_train_features,best_test_features=MDMR(train_feature, train_label, test_feature)

"""实例化RandomForestClassifier"""

estimator = RandomForestClassifier(n_jobs=-1,random_state=0,verbose=1)

"""-------------------------------------------------------------------------"""

"""n_estimators代表随机森林中决策树的数目"""

n_estimators=np.arange(20,200,20)
max_features=np.arange(1,20,2)
param_dict={"n_estimators":n_estimators,"max_features":max_features}

"""实例化GridSearchCV"""

estimator=GridSearchCV(estimator, param_grid=param_dict,cv=5)

"""-------------------------------------------------------------------------"""

estimator.fit(best_train_features,train_label)

print("最佳估计器: \n",estimator.best_estimator_)
print("最佳参数:\n",estimator.best_params_)

best_estimator=estimator.best_estimator_

"""-----------------------------十折交叉验证---------------------------------"""
k=10
np.random.seed(1234)

num=len(train_label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))

score_RF=np.zeros(num)
score_RF_prob=np.zeros(num)

for fold in range(k):
    
    trainLabel=train_label[mode!=fold]
    testLabel=train_label[mode==fold]
    
    trainFeature=best_train_features[mode!=fold]
    testFeature=best_train_features[mode==fold]
    
    best_estimator.fit(trainFeature,trainLabel)
    score_RF[mode==fold]=best_estimator.predict(testFeature)
    score_RF_prob[mode==fold]=best_estimator.predict_proba(testFeature)[:,1]

np.savez("data_all-model4-train.npz",predict=score_RF_prob,label=train_label)        
print("最佳估计器: \n",estimator.best_estimator_)
print("最佳参数:\n",estimator.best_params_)
        
print("Ten fold validation acc:%.3lf"%accuracy_score(train_label,score_RF))
print("Ten fold validation precision:%.3lf"%precision_score(train_label,score_RF))
print("Ten fold validation recall:%.3lf"%recall_score(train_label,score_RF))
print("Ten fold validation MCC:%.3lf"%matthews_corrcoef(train_label,score_RF))
print("Ten fold validation AUC:%.3lf"%roc_auc_score(train_label,score_RF_prob))
"""-------------------------------------------------------------------------"""

best_estimator.fit(best_train_features,train_label)
score=best_estimator.score(best_test_features,test_label)
test_predict=best_estimator.predict(best_test_features)
print("indepandent test acc：%.3lf"%score)
print("indepandent test precision：%.3lf"%precision_score(test_label,test_predict))
print("indepandent test recall：%.3lf"%recall_score(test_label,test_predict))
print("indepandent test MCC：%.3lf"%matthews_corrcoef(test_label,test_predict))

test_predict_pro=best_estimator.predict_proba(best_test_features)[:,1]

auc=roc_auc_score(test_label,test_predict_pro)
print("indepandent test AUC：%.3lf"%auc)
np.savez("data_all-model4-test.npz",predict=test_predict_pro,label=test_label)

    