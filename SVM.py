#!/usr/bin/env python
# coding: utf-8

# # <center> SVM (Support Vector Machine) quick look

# In[1]:


# Library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


# Simulate data

X_10 = np.random.normal(loc=0.0, scale=1.0, size=[500,2])
X_11 = np.random.normal(loc=7, scale=1.0, size=[500,2])
X_1 = np.append(X_10,X_11,axis=0)
X_2 = np.random.normal(loc=5, scale=2, size=[1000,2])


# In[3]:


Y_1 = np.zeros_like(X_1[:,0])
Y_2 = np.ones_like(X_2[:,0])


# In[4]:


# Data and target

X = np.append(X_1, X_2, axis=0)
Y = np.append(Y_1, Y_2, axis=0)


# In[5]:


# Plot data

plt.plot(X_1[:,0],X_1[:,1],'bo')
plt.plot(X_2[:,0],X_2[:,1],'ro')
plt.show()


# In[6]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[7]:


print("Shape X_train :", X_train.shape)
print("Shape Y_train :", Y_train.shape)
print("Shape X_test :", X_test.shape)
print("Shape Y_test :", Y_test.shape)


# In[8]:


model_svm = svm.SVC().fit(X_train, Y_train)


# In[9]:


predict_train = model_svm.predict(X_train)
print("Accuracy train : ", accuracy_score(Y_train, predict_train))


# In[10]:


predict_test = model_svm.predict(X_test)
print("Accuracy train : ", accuracy_score(Y_test, predict_test))


# ## <center> Compare with another model

# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


model_logistic_reg = LogisticRegression(random_state=0).fit(X_train, Y_train)


# In[13]:


predict_train_reg = model_logistic_reg.predict(X_train)
print("Accuracy train : ", accuracy_score(Y_train, predict_train_reg))


# In[14]:


predict_test_reg = model_logistic_reg.predict(X_test)
print("Accuracy test : ", accuracy_score(Y_test, predict_test_reg))


# <b> We can see that SVM has a better performance than logistic regression here. This can be explain by the fact that the data from each categories overlap each over <b>
