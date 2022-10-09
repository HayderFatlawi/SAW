# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 23:49:47 2022

@author: Hayder

"""
import pandas as pd 
import numpy as np
from collections import Counter
import time
import math
from skmultiflow.data import FileStream
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.evaluation import EvaluatePrequential
import socket, pickle
from sklearn.metrics import precision_score, accuracy_score
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt
import statistics

def ac(Y,YY):
    i=0
    cn=0
    cp=0
    ctp=0
    cfn=0
    ctn=0
    cfp=0
    tp=0
    tn=0
    fp=0
    fn=0
    acc=0
    prec=0
    while i<len(YY):
        if Y[i]==1:
            cp+=1
            if YY[i]==1:
                ctp+=1
            else:
                cfn+=1
        if Y[i]==0:
            cn+=1
            if YY[i]==1:
                cfp+=1
            else:
                ctn+=1
        i+=1
        
    if cp>0:
        if cp==ctp:
            tp=1
        elif cp!=0:
            tp=ctp/cp
            
     
        fn=cfn/cp
    # else:
    #     tp='No Positive'
        
    if cn>0:
        if cn==ctn:
            tn=1
        elif cn!=0:
            tn=ctn/cn
            
       
        fp=cfp/cn
    # else:
    #     tp='No Nagitive'
    
    acc=(ctp+ctn)/(cp+cn)
    
    if (ctp+cfp)!=0:
        prec=ctp/(ctp+cfp)
    
    return(tp,tn,fp,fn,acc,prec,cp)
    
#stream = FileStream("F:\\src\\All.csv")
stream =np.genfromtxt('F:\\src\\Siena EEG\\SienaR.csv',delimiter=',')

k=0
rm=0

n_samples = 1
max_samples = 160000
qw=1600
acc=[]
#ht=OzaBaggingClassifier(base_estimator=HoeffdingTreeClassifier(),n_estimators=20)
ht=AdaptiveRandomForestClassifier()
X, Y = stream[n_samples:n_samples+qw,:-1], stream[n_samples:n_samples+qw,-1]
R=np.ones((qw,), dtype=int)

main_c=0
tpar=[]
tnar=[]
fpar=[]
fnar=[]
accc=[]
precc=[]
start=time.time()
p1=0
p2=0
p3=0
p4=0
p5=0
p6=0
div=98
cp=0
ones_ar=[]
cp_ar=[]
while n_samples < max_samples:
  
    YY=ht.predict(X)
    #acc.append(accuracy_score(Y, YY))
    #acc.append(ac(Y, YY))
    tp_ar,tn_ar,fp_ar,fn_ar,accu,precis,cp=ac(Y, YY)
    #print(ac(Y, YY))
    if main_c >1:
        tpar.append(tp_ar)
        tnar.append(tn_ar)
        fpar.append(fp_ar)
        fnar.append(fn_ar)
        accc.append(accu)
        precc.append(precis)
        cp_ar.append(cp/len(Y))
        print(cp/len(Y))
        p1+=tp_ar
        p2+=tn_ar
        p3+=fp_ar
        p4+=fn_ar
        p5+=accu
        p6+=precis
        
    #prec=precision_score(Y, YY)
    ht.partial_fit(X, Y,[0,1])
    n_samples += qw
    
    
    # print('accaurcy:'+str(ac(Y, YY)))
    A, B= stream[n_samples:n_samples+qw,:-1], stream[n_samples:n_samples+qw,-1]
    Q=np.ones((qw,), dtype=int)
    
 
    
    
    T=np.empty_like(A)
    ind=0
    for b in B:
        if b==1:
            
            T=np.vstack((T,A[ind]))
        ind+=1
    
    Mn=np.mean(T,axis=0)
    
    
    
    
    v=np.count_nonzero(Y == 1)
    BB=[]
    ind=0
    v=0
    for y in Y:
          if y==1:
              BB.append(ind)
              v+=1
          ind+=1
    ones_ar.append(v)
        
    z=0
    Distance=[]
    summ=0
    #temp=random.sample(BB,v)
    temp=BB
    r=[]
    for t in temp:
        Distance.append(math.dist(X[t,:],Mn))
        r.append(t)
        R[t]+=1
        summ+=R[t]
        z+=1
   
    K=2+round(summ/z)
    
    Distance,temp=zip(*sorted(zip(Distance, temp)))
    
    MD=statistics.mean(Distance)
    v=0
    for d in Distance:
        if d<=MD:
            v+=1
    
    
    # print(v)
    # CC=np.where(B == 0)
    
    # BB=np.random.choice(CC[0],v,replace=False)
   
    # A =np.delete(A,BB,axis=0)
    # B=np.delete(B,BB,axis=0)
    

   
    
    for t,d in zip(temp,Distance):
        
        if  R[t]<=8:   
            C=np.vstack((A, X[t,:]))
            B=np.append(B, Y[t])   
            Q=np.append(Q, R[t]) 
            A=C
            
    
        
    X=A
    Y=B
    R=Q
  
  
    main_c+=1

print('tp:'+str(p1/div))
print('tn:'+str(p2/div))
print('fp:'+str(p3/div))
print('fn:'+str(p4/div))
print(p5/div)
print(p6/div)

print(tpar)
print(tnar)
print(fpar)
print(fnar)
print(accc)
print(precc)
print(ones_ar)
print(cp_ar)

