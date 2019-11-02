#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 14:47:54 2018

@author: justinpeter
"""


import pandas as pd
import numpy as np
from collections import Counter

iris = pd.read_csv('/Users/justinpeter/Documents/Data/iris.txt', header = None)

# Regression
X = iris.iloc[:, 0:3]
y = iris.iloc[:,3]

def NNR(X,y,z,k):
    n,p=X.shape
    distance=np.zeros((n))
    
    for i in range(n):
            distance[i]=np.sqrt(np.square(X.iloc[i,0]-z[0])+np.square(X.iloc[i,1]-z[1])+np.square(X.iloc[i,2]-z[2]))
            
    index_of_top_k_value=sorted(range(len(distance)),key=lambda i:distance[i])[:k]
    return np.mean(y[index_of_top_k_value])

z=[5.4,3.9,1.7]
NNR(X,y,z,k=5)

# Classification
X = iris.iloc[:, 0:4]
y = iris.iloc[:,4]

def NNC(X,y,z,k):
    n,p=X.shape
    distance=np.zeros((n))
    
    for i in range(n):
            distance[i]=np.sqrt(np.square(X.iloc[i,0]-z[0])+np.square(X.iloc[i,1]-z[1])+np.square(X.iloc[i,2]-z[2])+np.square(X.iloc[i,3]-z[3]))

    index_of_top_k_value = sorted(range(len(distance)), key=lambda i:distance[i])[:k]
    return Counter(list(y[index_of_top_k_value])).most_common(1)[0][0]

z = [4.9, 3.3, 1.5, 0.2]
X = iris.iloc[:, 0:4]
y = iris.iloc[:,4]
NNC(X, y, z, k=5)



# Problem 2
## Part (a)
X = [[0, 0, 1, 1, 0, -1], [1, 1, 0, 1, 0, -1],
[0, 1, 1, 1, 1, -1],
[1, 1, 1, 1, 0, -1],
[0, 1, 0, 0, 0, -1],
[1, 0, 1, 1, 1, 1],
[0, 0, 1, 0, 0, 1],
[1, 0, 0, 0, 0, 1],
[1, 0, 1, 1, 0, 1],
[1, 1, 1, 1, 1, -1]]

BP2=pd.DataFrame(X)
BP2.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'y']

X1_cond_table = pd.crosstab(BP2.X1 == 1, BP2.y > 0)
X2_cond_table = pd.crosstab(BP2.X2 == 1, BP2.y > 0)
X3_cond_table = pd.crosstab(BP2.X3 == 1, BP2.y > 0)
X4_cond_table = pd.crosstab(BP2.X4 == 1, BP2.y > 0)
X5_cond_table = pd.crosstab(BP2.X5 == 1, BP2.y > 0)


# calculate p(y)
prob_y_neg = np.sum(BP2.y == -1) / len(BP2.y) # p(y=-1) = 6/10
prob_y_pos = np.sum(BP2.y == 1)  / len(BP2.y) # p(y=1)  = 4/10
print("P(y=-1) is ", prob_y_neg)
print("P(y=+1) is ", prob_y_pos)

# calculate P(X|y)
BP2.groupby('y')['X1'].value_counts()/BP2.groupby('y')['X1'].count()
BP2.groupby('y')['X2'].value_counts()/BP2.groupby('y')['X2'].count()
BP2.groupby('y')['X3'].value_counts()/BP2.groupby('y')['X3'].count()
BP2.groupby('y')['X4'].value_counts()/BP2.groupby('y')['X4'].count()
BP2.groupby('y')['X5'].value_counts()/BP2.groupby('y')['X5'].count()


# part (b) 
# section (a) predict class for x=(0 0 0 0 0) and x=(1 1 0 1 0)
prob_y_1_given_x_all_0 = (1/4)*1*(1/4)*(2/4)*(3/4)*(4/10) / ((3/6)*(5/6)*(4/6)*(5/6)*(2/6)*(6/10))
print("P(y=1|x=(0,0,0,0,0)=", prob_y_1_given_x_all_0)
prob_y_1_given_x_some_0 = (3/4)*(0)*(1/4)*(2/4)*(3/4)*(4/10) / ((3/6)*(5/6)*(4/6)*(5/6)*(2/6)*(6/10))
print("P(y=1|x=(1,1,0,1,0)=", prob_y_1_given_x_some_0)


# section (b)
print("We probably do not have to use a joint Bayes classifier as opposed to a naive \
classifier for this data because we assume that the variables are independent. The model \
regarding naive Bayes classifier has this assumption of independent attributes whereas \
the joint Bayes classifier assumes that the attributes are not independent.") 


# section (c)
Y=[[0, 1, 1, 0, -1], [1, 0, 1, 0, -1],
[1, 1, 1, 1, -1],
[1, 1, 1, 0, -1],
[1, 0, 0, 0, -1],
[0, 1, 1, 1, 1],
[0, 1, 0, 0, 1],
[0, 0, 0, 0, 1],
[0, 1, 1, 0, 1],
[1, 1, 1, 1, -1]]

BP3=pd.DataFrame(Y)

BP3.columns = ['X2', 'X3', 'X4', 'X5', 'y']
BP3.groupby('y')['X2'].value_counts()/BP3.groupby('y')['X2'].count()
BP3.groupby('y')['X3'].value_counts()/BP3.groupby('y')['X3'].count()
BP3.groupby('y')['X4'].value_counts()/BP3.groupby('y')['X4'].count()
BP3.groupby('y')['X5'].value_counts()/BP3.groupby('y')['X5'].count()


prob_y_1_given_x2toX5 = (1/4)*(2/4)*(3/4)*(4/10) / ((5/6)*(4/6)*(5/6)*(2/6)*(6/10))
print("P(y=1)|x=(0,0,0,0)=", prob_y_1_given_x2toX5)

print("We need to re-train the model in order to update the posterior probabilities with the extra constraint \
      that attribute X1 is removed. When we compute the new posterior probabilities, \
        we observed that the probability increased as soon as the attribute X1 was removed from the data set.")