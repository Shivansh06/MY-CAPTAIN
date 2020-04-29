#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


# In[8]:


boston=load_boston()


# In[9]:


boston.keys()


# In[13]:


print("The target values are :\n{}".format(boston['target']))


# In[26]:


print(boston['target'].shape)


# In[16]:


print("The feature names are :\n{}".format(boston['feature_names']))


# In[17]:


print(boston['DESCR'])


# In[19]:


print("THE DATA IS :\n{}".format(boston['data']))


# In[20]:


print(type(boston['data']))


# In[22]:


df=boston['data']


# In[24]:


print(df.shape)


# In[66]:


get_ipython().system('matplotlib inline')
plt.scatter(df[:,0],boston['target'],color='green')
plt.xlabel("Crime Rates")
plt.ylabel("Price of the house")


# In[67]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(df[:,0].reshape(-1,1),boston['target'])


# In[68]:


pred=reg.predict(df[:,0].reshape(-1,1))


# In[70]:


get_ipython().system('matplotlib inline')
plt.scatter(df[:,0],boston['target'],color='green')
plt.plot(df[:,0],pred,color='red')
plt.xlabel("Crime rates")
plt.ylabel("Price of the house")
plt.show()


# In[138]:


#circumventing curve issue using polynomial
from sklearn.preprocessing import PolynomialFeatures


# In[145]:


#to allow merging of models 
list1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
scores=[]
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
for k in list1:
    model=make_pipeline(PolynomialFeatures(k),reg)
    model.fit(df[:,0].reshape(-1,1),boston['target'])
    pred1=model.predict(df[:,0].reshape(-1,1))
    score2=r2_score(pred1,boston['target']) 
    scores.append(score2)
zipped=zip(list1,scores)
best=[(x,y) for (x,y) in zipped if y==max(scores)]
print(best)


# In[146]:


#the above shows that we get the best score when power of the polynomial function is 8
model=make_pipeline(PolynomialFeatures(8),reg)
model.fit(df[:,0].reshape(-1,1),boston['target'])
pred1=model.predict(df[:,0].reshape(-1,1))
get_ipython().system('matplotlib inline')
plt.scatter(df[:,0],boston['target'],color='green')
plt.plot(df[:,0],pred1,color='red')
plt.show()


# In[149]:


print("The score for the polynomial is {}".format(best[0][1]))
print("The score for linear regression is {}".format(r2_score(pred,boston['target'])))


# In[ ]:




