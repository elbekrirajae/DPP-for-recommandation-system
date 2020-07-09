# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:33:59 2020

@author: User
"""

from pydpp.dpp import DPP
import numpy as np

nb_iter=10
X=[]
Matrix=[]
for i in range(10):
    x = [np.random.randint(1, 10) for p in range(0, 50)]
    X.append(x)
X=np.asarray(X)
    
dpp = DPP(X)
dpp.compute_kernel(kernel_type='cos-sim')
idx = dpp.sample_k(5)

for k in range(nb_iter):

    for j in range(10):
        ran=np.random.randint(1,10)
        idx = dpp.sample_k(ran)
    
        Matrix.append(X[idx,j])
        
    k+=1
