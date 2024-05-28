import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# %reload_ext autoreload
from os import system
import subprocess
import numpy as np
import itertools as itt
import multiprocessing
import time
import math
# import de_nn as de_nn
from scipy.stats import qmc
from numpy import *
import random
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
#import mxnet.ndarray as nd
#import mxnet as mx
from itertools import chain
import time
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
# model1=load_model('model_3combined.h5')
# model1=load_model('model_combined2.h5')
model1=load_model('model_combined.h5')

def polar_to_cartesian(pop):
    rad = pop[:,0:8]
    phi = pop[:,8:16]
    alp = pop[:,16]

    tht = np.zeros((np.shape(pop)[0],8))

    phi1 = np.cumsum(phi, axis=1)
    # print(alp, '\n', phi1)
    for i in range (np.shape(pop)[0]):
        for j in range(8):
            tht[i,j] = phi1[i,j]*alp[i]/phi1[i,-1]
    
    X=[]
    Y=[]
    for i in range(np.shape(pop)[0]):
        r, t = rad[i], tht[i]
        xc = np.zeros(8)
        yc = np.zeros(8)

        for j in range (8):
            xc[j] = r[j]*math.cos(t[j])
            yc[j] = r[j]*math.sin(t[j])

        X.append(xc/1000)
        Y.append(yc/1000)
    XY=np.concatenate((X,Y),axis=1)
    return (XY)


def prediction(pop):
    xy = polar_to_cartesian(pop)
    lower = np.array([-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,
                  -0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03])
                  

    upper = np.array([0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,
                  0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03])
                  
    for i in range (np.shape(xy)[0]):
        xy[i] = (xy[i]-lower)/(upper-lower)

    p = model1.predict(xy)
    return(p)


def get_mse(pop, tar,n_i,n_p):
    isl = np.array(list(chain.from_iterable(pop)))
    tar = np.tile(tar, (n_i*n_p,1))    
    pp = prediction(isl)
    mse = pp[:,52]
#     mse = np.mean((tar - pp)**2, axis=1)
    return np.reshape(mse,(n_i,n_p))

def get_mse3(pop, tar,n_i,n_p):
    isl = np.array(list(chain.from_iterable(pop)))
    Gtar = tar[0:41600]
    Gtar = np.reshape(Gtar,(128,25,13))
    Gtar = Gtar[77,:,:]
    Star = tar[41600:41728]
    Rtar = tar[41728:41856]
    
    tar1 = np.tile(Gtar, (n_i*n_p,1,1,1))
    tar2 = np.tile(Star, (n_i*n_p,1))
    tar3 = np.tile(Rtar, (n_i*n_p,1))
    
    y = prediction(isl)
    y1=y[:,0:41600]
    y1=np.reshape(y1,(n_i*n_p,13,25,128))
    y1=np.swapaxes(y1,1,3)
    y1=y1[:,74:79,:,:]

    y2=y[:,41600:41728]
    y3=y[:,41728:41856]
#     print(np.shape(Gtar),np.shape(tar1),np.shape(y1))
    m_1 = np.mean(np.mean(np.mean((y1-tar1)**2, axis = 3), axis = 2), axis =1)
#     m_1 = np.reshape(m_1,(n_i*n_p,325))
    m_2 = np.mean((y2-tar2)**2, axis = 1)
    m_3 = np.mean((y3-tar3)**2, axis = 1)
#     print(np.shape(m_1),np.shape(m_2))
    mse = 0.5*m_1 + m_2 + 0.01*m_3
#     print(m)
    return np.reshape(mse,(n_i,n_p))
