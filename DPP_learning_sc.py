# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 00:11:55 2020

@author: User
"""


#%%
import numpy as np
from pydpp.dpp import DPP
import numpy.linalg as lg
import math
import random
#%%
def computeLogLikeLihood(LowRankMatrix,nb_samples,nb_items,nb_trait,training_samples,lambdaVec,alpha):
    #first term of the LL
    L=LowRankMatrix@LowRankMatrix.T
    sumDetFirstTerm=0
    Log_Determinant_LN=0
    for i in range(nb_samples):
        V_n=np.take(LowRankMatrix,training_samples[i], axis=0)
        #V_n=LowRankMatrix[training_samples[i],:]
        #print(V_n.shape)
        L_n=V_n@V_n.T
        #print(L_n.shape[0])
        if L_n.shape[0] ==1 :
            determinant_Ln=0
        else :
            determinant_Ln=lg.det(L_n)
        if determinant_Ln>0:
            Log_Determinant_LN=math.log(determinant_Ln)
        sumDetFirstTerm+=Log_Determinant_LN
    firstTerm=sumDetFirstTerm
    #second term of LL
    DetSecTerm=lg.det(np.identity(nb_items)+L)
    if DetSecTerm>0:
        LogSecTerm=math.log(DetSecTerm)
        
    SecondTerm=nb_samples*LogSecTerm
    
    #Third term ( regularization)

    Third1=0
    for i in range(nb_items):
        #print(lambdaVec[i])
        lmbd=lambdaVec[i]
        norm=lg.norm(LowRankMatrix[i,:], 2)
        thirdd=np.round(lmbd,5)*np.round(norm,5)
        Third1+= thirdd
    ThirdTerm=0
    if alpha!=0:
        ThirdTerm=alpha*Third1
    
    #compute the LL
    
    LL=firstTerm-SecondTerm-0.5*ThirdTerm
    
    return LL
#%%
    
def computeGradient(LowRankMatrix,nb_samples,nb_items,nb_trait,training_samples,lambdaVec,alpha):
    M,K=LowRankMatrix.shape #utiliser nb_items et nb_trait !
    sumTraceFirstTermGradient=0
    Gradient_Matrix=np.zeros((M,K))
    VMatrix_NinstanceVector=[]
    LMatrix_Ninstance_InverseVector=[]
    Size_V_NinstanceVector=[]
    #Martix of all V[samples] and L[samples]inverse
    #first term
    for i in range(nb_samples):
        V_Ninstance=LowRankMatrix[training_samples[i],:]
        VMatrix_NinstanceVector.append(LowRankMatrix[training_samples[i],:])
        
        LMatrix_Ninstance=V_Ninstance@V_Ninstance.T
        
        LMatrix_Ninstance_Inverse=lg.pinv(LMatrix_Ninstance)
        
        LMatrix_Ninstance_InverseVector.append(LMatrix_Ninstance_Inverse)
        
        Size_V_NinstanceVector.append(V_Ninstance.shape[0])
    
    
    #second term
    pre_secondTerm=np.identity(M)-LowRankMatrix@(lg.pinv(np.identity(K)+(LowRankMatrix.T)@LowRankMatrix))@LowRankMatrix.T
    
      
    ## compute the gradient for F(i,k)
    BuildMAP=Build_Map_Training_RowCol(training_samples,nb_items,nb_samples)
    for k in range(K):
        for i in range(M):
            SumTraceFirstTerm=0
            SumTraceSecondTerm=0
            for l in range(nb_samples):
                A=LMatrix_Ninstance_InverseVector[l]
                V=VMatrix_NinstanceVector[l]
                size=Size_V_NinstanceVector[l]
                instance=BuildMAP[l,i]
                if instance !=0 :
                    itemNotPresent=False
                else :
                    itemNotPresent=True
                traceFirst=0
                if itemNotPresent:
                    traceFirst=0
                else :
                    sumFirst=0
                    for j in range(size):
                        sumFirst+=A[j,int(instance)]*V[j,k]
                    traceFirst=sumFirst + A[int(instance),:]@V[:,k]
                
                SumTraceFirstTerm+=traceFirst
                
            
            SumAdotV2=0
            B=pre_secondTerm
            for j in range(M):
                SumAdotV2+=B[j,i]*LowRankMatrix[j,k]
            
            SumTraceSecondTerm=B[i,:]@LowRankMatrix[:,k]+SumAdotV2
            
            ##gradient a l indice i,k
            FinalTerm=SumTraceFirstTerm-nb_samples*SumTraceSecondTerm-alpha*lambdaVec[i]*LowRankMatrix[i,k] 
            
            Gradient_Matrix[i,k]=FinalTerm
            
    return Gradient_Matrix
#%%
def StochasticGradientAscent(training_samples,nb_samples,nb_items,nb_trait,max_iteration,lambdaVec,alpha):
    M,K=nb_items,nb_trait
    gradient=np.zeros((M,k))
    delta=np.zeros((M,K))
    pourc_test=0.3
    pourc_valid=0.1
    Eps0=1e-5
    T=60
    Beta=0.95
    counter=0
    Epsilon=Eps0/(1+counter/T)
    split_train=int(nb_samples*0.7)
    split_valid=int(nb_samples*0.1)
    InitialParam=np.random.uniform(0,1,(M,K))
    minibatch=20
    curr_index=0
    #training=np.random.shuffle(training_samples)
    #traininf=map(numpy.random.shuffle, training_samples)
    training=random.sample(training_samples,nb_samples)

    Train=training[:split_train]
    Valid=training[split_train:split_train+split_valid]
    Test=training[split_train+split_valid:]
    nb_train=len(Train)
    nb_test=len(Test)
    nb_valid=len(Valid)
    LowRankMatrix=InitialParam
    valid_LL_first=computeLogLikeLihood(LowRankMatrix,nb_valid,nb_items,nb_trait,Valid,lambdaVec,alpha)
    #LogLike_valid=0
    while counter< max_iteration :
        
        
        nb_instance_Batch=minibatch
        
        if curr_index+minibatch> nb_train :
            nb_instance_Batch=nb_train-curr_index
        
        train_batch=Train[curr_index:curr_index+minibatch]
        
        
        gradient=computeGradient(LowRankMatrix+Beta*delta,nb_instance_Batch,M,K,train_batch,lambdaVec,alpha)
        
        delta=Beta*delta + (1-Beta)*Epsilon*gradient 
        
        LowRankMatrix=LowRankMatrix+delta
        
        LL_train=computeLogLikeLihood(LowRankMatrix,nb_train,M,K,Train,lambdaVec,alpha)
        
        LL_valid=computeLogLikeLihood(LowRankMatrix,nb_valid,M,K,Valid,lambdaVec,alpha)

        LL_test=computeLogLikeLihood(LowRankMatrix,nb_test,M,K,Test,lambdaVec,alpha)
                
        print("LogLikelihood for the training",LL_train)
        print("LogLikelihood for the test",LL_test)
        print("LogLikelihood for the valid",LL_valid)
        print(counter)
        counter+=1

        
        if np.abs(valid_LL_first-LL_valid) < 1e-6:
            break
        else :
            valid_LL_first=LL_valid
        
        curr_index+=nb_instance_Batch +1
        
        if curr_index> nb_train:
            #wwe processed all the samples, start from the beginning
            curr_index=1
            Train=random.sample(Train,nb_train)

            
        
    return LowRankMatrix
#%%
def compute_LambdaVec(training_samples,nb_samples,nb_items):
    lambdaVec=np.zeros((10,1))
    for i in range(nb_items):
        counter=0
        item=i
        for j in range(nb_samples):
            sample=training_samples[j]
            for k in range(len(sample)):
                if item==sample[k]:
                    counter+=1
        try:            
            lambdaVec[i]=1/counter
        except ZeroDivisionError:
            lambdaVec[i]=0
    vec=[]
    for lm in lambdaVec:
        vec.append(float(lm))
    return vec
#%%
def Build_Map_Training_RowCol(training_samples,nb_items,nb_samples):
    Build_Map=np.zeros((nb_samples,nb_items))
    for i in range(nb_samples):
        trainingInstanceItems=training_samples[i]
        
        for j in range(len(trainingInstanceItems)):
            
            Build_Map[i,trainingInstanceItems[j]]=j
    
    return Build_Map 
#%%
nb_iter=10000
X=[]
Matrix=[]
for i in range(10):
    x = [np.random.randint(0, 9) for p in range(0, 50)]
    X.append(x)
X=np.asarray(X)
    
dpp = DPP(X)
dpp.compute_kernel(kernel_type='cos-sim')
idx = dpp.sample_k(5)

for k in range(nb_iter):

    for j in range(10):
        ran=np.random.randint(2,10)
        idx = dpp.sample_k(ran)
    
        Matrix.append(X[idx,j])
        
    k+=1    

print(Matrix[0:10])

#dpp2=DPP(np.asarray(Matrix))

#%%
##########GENERATION DE DONNES###
#################################
training_samples=Matrix
nb_items=10
nb_trait=5
M,K=10,5
nb_samples=100000 #len(Matrix)
vec=compute_LambdaVec(training_samples,nb_samples,nb_items)
alpha=0.1

#%%
ma=StochasticGradientAscent(training_samples,100,10,5,100000,vec,alpha=0.1)

#%%

L_final=ma@ma.T
print(L_final)

#%%

L_dpp=dpp.A
print(L_dpp)

#%%

np.abs(L_dpp-L_final)

