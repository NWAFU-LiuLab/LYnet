# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:49:29 2022

@author: lvyang
"""
def MDMR(train_features,train_label,test_features):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.metrics import auc
    #step1:特征命名
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
            DV+=EDij
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
    
    rank_list=list(rank)
    """对特征名字也进行排序与train_features_rank对应"""
    feature_name_rank=[feature_name[x] for x in rank_list]
    
    #step3：用随机森林剔除冗余数据，寻找最高的AUC指标的特征组合
    MDMR_train_data,MDMR_test_data,MDMR_train_label,MDMR_test_label=train_test_split(train_features_rank,
                                                                                     train_label,test_size=0.2,
                                                                                     random_state=3,stratify=train_label)
    
    # 运行随机森林，寻找最高的AUC指标的特征组合，即 max_key的值
    conclusion = {}#创建检索字典
    
    for cnt in range(train_features_rank.shape[1]):
        
        X_train = MDMR_train_data[..., 0:cnt+1]#取前cnt项
        X_test = MDMR_test_data[..., 0:cnt+1]#取前cnt项
        
        clf = RandomForestClassifier(random_state=0,n_jobs=-1,verbose=1)
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