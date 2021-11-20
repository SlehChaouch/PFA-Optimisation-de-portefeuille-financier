# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
 
import pandas_datareader.data as web
import numpy as np
from math import sqrt
import statistics 
from statistics import mean
from numpy import linalg as LA
from numpy.linalg import inv
from scipy import stats
import matplotlib.pyplot as plt


data=pd.read_excel(r'C:\Users\HP\Desktop\2AMIndS\PFA\tunisia_stock_market.xlsx')
df=pd.DataFrame(data)
#for x in TickerDax : 
#   df2=web.DataReader(x, data_source='yahoo',start='2017-01-1',end='2017-12-31')
 #  df[x]=df2['Adj Close']
print(df) 


on=np.ones(5)
R=(df-df.iloc[0])/df.iloc[0]
R=R.iloc[1:255]
print(R)

M=np.zeros(5)
j=0
for i in df :
     M[j]=R[i].mean()
     j+=1
M=M*255     
print(M)

covMatrix=R.cov()
print(covMatrix)

ppl=np.dot(LA.inv(covMatrix),(np.transpose(on)))
a=np.dot(ppl,on)
print(a)

b=np.dot(ppl,M)
print(b)
t=np.arange(1/sqrt(a),0.5,0.001)
mu=np.zeros(len(t))
for i in range (len(t)) :
    mu[i]=b/a+sqrt((t[i]**2)-1/a)*sqrt(np.dot(np.linalg.inv(covMatrix)@(M-(b/a)*on),M-(b/a)*on))
# Matplotlib only plot: 
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(t**2, mu) #risks and returns are just arrays of points along the frontier
plt.show()

    
P=np.zeros(5)
I=np.zeros(5)
p=0
for i in df :
    P[p]=stats.jarque_bera(R[i]).statistic
    I[p]=stats.jarque_bera(R[i]).pvalue
    p+=1
print(P)
print(I)    
h=20
c=np.arange(0,0.5,0.001)
m=np.zeros(len(c))
for i in range(len(c)) :
    m[i] = h + sqrt(np.dot(np.linalg.inv(covMatrix)@(M-h*on),M-h*on))*c[i]
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(c,m)
plt.show()
s=0.05
pi=np.dot(np.linalg.inv(covMatrix),on)/a
pi+=(sqrt((s**2)-1/a)*np.dot(np.linalg.inv(covMatrix),(M-(b/a)*on)))/(sqrt(np.dot(np.linalg.inv(covMatrix)@(M-(b/a)*on),M-(b/a)*on)))
print(pi)