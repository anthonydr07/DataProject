#!/usr/bin/env python
# coding: utf-8

# # <center>  ACP and T-SNE

# In[1]:


# Library

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal  
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ## PCA

# In[2]:


# Gaussian data simulation

n = 100
mean = np.array([0 for i in range(n)])
cov = np.eye(n) 
col = ['X'+str(i) for i in range(n)]
X_1 = pd.DataFrame(np.random.multivariate_normal(mean, cov, (500)),columns=col)
X_1['Class'] = 0
X_2 = pd.DataFrame(np.random.multivariate_normal(mean+3, cov+3, (500)),columns=col)
X_2['Class'] = 1
X = X_1.append(X_2).reset_index(drop=True)


# In[3]:


# Data
X.head()


# In[4]:


# Center and reduice data
X_center = (X.iloc[:,:100] - X.iloc[:,:100].mean())/X.iloc[:,:100].var()


# In[5]:


X_center['Class'] = X['Class']


# In[6]:


# PCA with 2 components
n_components = 2
pca = PCA(n_components=n_components,random_state=42)
pca_fit = pca.fit(X_center.iloc[:,:100])
print(f"var explained : {np.sum(pca_fit.explained_variance_ratio_)}")
pca_trans = pca.fit_transform(X_center)
pca_trans


# In[7]:


# Plot result
plt.scatter(x=pca_trans[:500,:1],y=pca_trans[:500,1:2],c="red")
plt.scatter(x=pca_trans[500:,:1],y=pca_trans[500:,1:2],c="blue")


# In[8]:


# Search for the best number of components based on the explained variance
var_explained = []
n_comp = []
for i in range(2,50):
    n_components = i
    pca_i = PCA(n_components=n_components,random_state=42)
    pca_fit_i = pca_i.fit(X_center.iloc[:,:100])
    var_explained.append(np.sum(pca_fit_i.explained_variance_ratio_))
    n_comp.append(i)


# In[9]:


plt.plot(n_comp,var_explained)


# ## t-SNE

# In[10]:


tsne = TSNE(n_components=2)


# In[11]:


tsne_fit = tsne.fit_transform(X.iloc[:,:100])


# In[12]:


tsne_fit_center = tsne.fit_transform(X_center.iloc[:,:100])
plt.scatter(x=tsne_fit_center[:500,:1],y=tsne_fit_center[:500,1:2],c="red")
plt.scatter(x=tsne_fit_center[500:,:1],y=tsne_fit_center[500:,1:2],c="blue")

