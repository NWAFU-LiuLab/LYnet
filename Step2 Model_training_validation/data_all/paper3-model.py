# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:57:29 2022

@author: lvyang
"""

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

model = QuadraticDiscriminantAnalysis()

def features(file):
    f=open(file,"r")
    aaindex_values=[]
    lines=f.readlines()
    
    for sequence in lines[1::2]:

        GRAR740102={'A':8.1,'L':4.9,'R':10.5,'K':11.3,'N':11.6,
                'M':5.7,'D':13.0,'F':5.2,'C':5.5,'P':8.0,
                'Q':10.5,'S':9.2,'E':12.3,'T':8.6,'G':9.0,
                'W':5.4,'H':10.4,'Y':6.2,'I':5.2,'V':5.9}
        
        WOEC730101={'A':7.0,'L':4.9,'R':9.1,'K':10.1,'N':10.0,
                'M':5.3,'D':13.0,'F':5.0,'C':5.5,'P':6.6,
                'Q':8.6,'S':7.5,'E':12.5,'T':6.6,'G':7.9,
                'W':5.3,'H':8.4,'Y':5.7,'I':4.9,'V':5.6}
        
        KIDA850101={'A':-0.27,'L':-1.10,'R':1.87,'K':1.70,'N':0.81,
                'M':-0.73,'D':0.81,'F':-1.43,'C':-1.05,'P':-0.75,
                'Q':1.10,'S':0.42,'E':1.17,'T':0.63,'G':-0.16,
                'W':-1.57,'H':0.28,'Y':-0.56,'I':-0.77,'V':-0.40}
        
        QIAN880108={'A':0.33,'L':0.57,'R':0.10,'K':0.23,'N':-0.19,
                'M':0.79,'D':-0.44,'F':0.48,'C':-0.03,'P':-1.86,
                'Q':0.19,'S':-0.23,'E':0.21,'T':-0.33,'G':-0.46,
                'W':0.15,'H':0.27,'Y':-0.19,'I':-0.33,'V':0.24}
        
        FAUJ830101={'A':0.31,'L':1.70,'R':-1.01,'K':-0.99,'N':-0.60,
                'M':1.23,'D':-0.77,'F':1.79,'C':1.54,'P':0.72,
                'Q':-0.22,'S':-0.04,'E':-0.64,'T':0.26,'G':0.00,
                'W':2.25,'H':0.13,'Y':0.96,'I':1.80,'V':1.22}
        
        PONP800103={'A':2.63,'L':2.98,'R':2.45,'K':2.12,'N':2.27,
                'M':3.18,'D':2.29,'F':3.02,'C':3.36,'P':2.46,
                'Q':2.45,'S':2.60,'E':2.31,'T':2.55,'G':2.55,
                'W':2.85,'H':2.57,'Y':2.79,'I':3.08,'V':3.21}
        
        LAWE840101={'A':-0.48,'L':1.02,'R':-0.06,'K':-0.09,'N':-0.87,
                'M':0.81,'D':-0.75,'F':1.03,'C':-0.32,'P':2.03,
                'Q':-0.32,'S':0.05,'E':-0.71,'T':-0.35,'G':0.00,
                'W':0.66,'H':-0.51,'Y':1.24,'I':0.81,'V':0.56}
        
        MIYS990105={'A':-0.02,'L':-0.32,'R':0.08,'K':0.30,'N':0.10,
                'M':-0.25,'D':0.19,'F':-0.33,'C':-0.32,'P':0.11,
                'Q':0.15,'S':0.11,'E':0.21,'T':0.05,'G':-0.02,
                'W':-0.27,'H':-0.02,'Y':-0.23,'I':-0.28,'V':-0.23}
        
        EISD860103={'A':0.0,'L':0.89,'R':-0.96,'K':-0.99,'N':-0.86,
                'M':0.94,'D':-0.98,'F':0.92,'C':0.76,'P':0.22,
                'Q':-1.0,'S':-0.67,'E':-0.89,'T':0.09,'G':0.0,
                'W':0.67,'H':-0.75,'Y':-0.93,'I':0.99,'V':0.84}
        
        ARGP820102={'A':1.18,'L':3.23,'R':0.20,'K':0.06,'N':0.23,
                'M':2.67,'D':0.05,'F':1.96,'C':1.89,'P':0.76,
                'Q':0.72,'S':0.97,'E':0.11,'T':0.84,'G':0.49,
                'W':0.77,'H':0.31,'Y':0.39,'I':1.45,'V':1.08}
        
        aaindex_list = [GRAR740102,WOEC730101,KIDA850101,QIAN880108,FAUJ830101,PONP800103,LAWE840101,MIYS990105,EISD860103,ARGP820102]
        aaindex_valuesT=[]    
        for i in aaindex_list:
            a_a = ((sequence.count("A") * i["A"])) / len(sequence)
            c_c = ((sequence.count("C") * i["C"])) / len(sequence)
            d_d = ((sequence.count("D") * i["D"])) / len(sequence)
            e_e = ((sequence.count("E") * i["E"])) / len(sequence)
            f_f = ((sequence.count("F") * i["F"])) / len(sequence)
            g_g = ((sequence.count("G") * i["G"])) / len(sequence)
            h_h = ((sequence.count("H") * i["H"])) / len(sequence)
            i_i = ((sequence.count("I") * i["I"])) / len(sequence)
            k_k = ((sequence.count("K") * i["K"])) / len(sequence)
            l_l = ((sequence.count("L") * i["L"])) / len(sequence)
            m_m = ((sequence.count("M") * i["M"])) / len(sequence)
            n_n = ((sequence.count("N") * i["N"])) / len(sequence)
            p_p = ((sequence.count("P") * i["P"])) / len(sequence)
            q_q = ((sequence.count("Q") * i["Q"])) / len(sequence)
            r_r = ((sequence.count("R") * i["R"])) / len(sequence)
            s_s = ((sequence.count("S") * i["S"])) / len(sequence)
            t_t = ((sequence.count("T") * i["T"])) / len(sequence)
            v_v = ((sequence.count("V") * i["V"])) / len(sequence)
            w_w = ((sequence.count("W") * i["W"])) / len(sequence)
            y_y = ((sequence.count("Y") * i["Y"])) / len(sequence)
    
            aaindex_comp = round(((a_a + c_c + d_d + e_e + f_f + g_g + h_h + i_i + k_k + l_l + m_m + n_n + p_p + q_q + r_r + s_s + t_t + v_v + w_w + y_y) / 20),3)
    
            aaindex_valuesT.append(aaindex_comp)
        aaindex_values.append(aaindex_valuesT)
    f.close()
    return np.array(aaindex_values)


data_positive=features("positive.fasta")
data_negative=features("negative.fasta")
data_train=features("Train.fasta")
data_test=features("Test.fasta")



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
train_label=get_label("Train.FASTA")
test_label=get_label("Test.FASTA")

encoding=np.concatenate((data_positive,data_negative,data_train,data_test),axis=0)
label=np.concatenate((np.ones(592),np.zeros(592),train_label,test_label),axis=0)

scale=MinMaxScaler()
encoding=scale.fit_transform(encoding)

train_encoding,test_encoding,train_label,test_label=train_test_split(encoding,label,train_size=0.9,random_state=22)


"""-----------------------------十折交叉验证---------------------------------"""
k=10
np.random.seed(1234)

num=len(train_label)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))

score=np.zeros(num)
score_prob=np.zeros(num)

for fold in range(k):
    
    trainLabel=train_label[mode!=fold]
    testLabel=train_label[mode==fold]
    
    trainFeature=train_encoding[mode!=fold]
    testFeature=train_encoding[mode==fold]
    
    model.fit(trainFeature,trainLabel)
    score[mode==fold]=model.predict(testFeature)
    score_prob[mode==fold]=model.predict_proba(testFeature)[:,1]
    
    
np.savez("data_all-model3-train.npz",predict=score_prob,label=train_label)            
acc=accuracy_score(train_label,score)
auc=roc_auc_score(train_label,score_prob)
recall=recall_score(train_label,score)
precision=precision_score(train_label,score)
f1=f1_score(train_label,score)
kappa=cohen_kappa_score(train_label,score)
mcc=matthews_corrcoef(train_label,score)

print(f"ACC:{acc}\nAUC:{auc}\nRecall:{recall}\nPrecision:{precision}\nF1:{f1}\nKappa:{kappa}\nMCC:{mcc}\n")




model.fit(train_encoding,train_label)
test_predict=model.predict(test_encoding)
test_predict_pro=model.predict_proba(test_encoding)[:,1]

acc=accuracy_score(test_label,test_predict)
auc=roc_auc_score(test_label,test_predict_pro)
recall=recall_score(test_label,test_predict)
precision=precision_score(test_label,test_predict)
f1=f1_score(test_label,test_predict)
kappa=cohen_kappa_score(test_label,test_predict)
mcc=matthews_corrcoef(test_label,test_predict)

print(f"ACC:{acc}\nAUC:{auc}\nRecall:{recall}\nPrecision:{precision}\nF1:{f1}\nKappa:{kappa}\nMCC:{mcc}")
np.savez("data_all-model3-test.npz",predict=test_predict_pro,label=test_label)        


