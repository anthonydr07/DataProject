#!/usr/bin/env python
# coding: utf-8

# # <center> Quick gradient descent with pytorch

# In[1]:


# Library

import torch
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Simulate data

x = np.arange(1,100)
a,b,c,d = 0,0,1,0
y = ((a*x)**3)+((b*x)**2)+(c*x)+d


# In[3]:


plt.plot(x,y)


# gradient descent
# 
# search function h such that h(x)≈y
# 
# params_new = params_old-delta*deriv

# In[4]:


# Center and reduice data

x_center = (x-x.mean())/x.std()
y_center = (y-y.mean())/y.std()


# In[5]:


# Gradient descent

def loss(y,y_pred):
    """
    Calculate the loss
    Params:
        y : target
        y_pred : predicted target
    Returns:
        Loss as a torch tensor
    """
    return torch.sum((y-y_pred)**2)

def h(x,params):
    """
    Calculate the predicted target based on the parameters
    Params:
        x : variables
        params : list of parameters
    Returns:
        list of predicted target
    """
    return ((params[0]*x)**3)+((params[1]*x)**2)+(params[2]*x)+params[3]

def gradient_descent(x,y,alpha=1e-3,n=1000,init_params=[0.1,0.1,0.1,0.1],print_loss=100):
    """
    Gradient descent
    Params:
        x : variables
        y : target
        alpha : learning rate
        n : number of iterations
        init_params : list of the initialized parameters
        print_loss : print the loss every i iteration
    Returns:
        list of the loss and list the best parameters
    """
    x_lenght = x.shape[0]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    error = loss(y,h(x,init_params))
    params = torch.tensor(init_params,requires_grad=True)
    optimizer = torch.optim.Adam([params],lr=alpha)
    losses = [error]
    to_stop=0
    
    while losses[-1]>1e-6 and to_stop<n:
        optimizer.zero_grad()
        for i in range(x_lenght):
            optimizer.step()
            error = loss(y[i],h(x[i],params))
            error.backward()
        losses.append(error.detach().numpy())
        to_stop+=1
        if to_stop%print_loss==0:
            print(f'loss : {error.detach().numpy()}')
            
    return losses,params.detach().numpy()


# In[6]:


error, params = gradient_descent(x,y,alpha=1e-3,n=1000,print_loss=100)


# In[7]:


plt.plot(error)


# In[8]:


h(x,params)


# In[ ]:




