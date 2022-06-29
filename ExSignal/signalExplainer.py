import tensorflow as tf

from tensorflow import keras
from keras.utils import np_utils
import shap


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
from scipy.stats import kurtosis
from scipy import signal
from sklearn.model_selection import train_test_split
from  sklearn.utils import shuffle
from os import path, mkdir
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import itertools
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import urllib.request

from keras.models import load_model

import komm
import sys
import glob
import os

from numpy import load

model_convlstm=load_model('model_convlstm.h5')
model_convlstm.summary()



dict_data = load('xFake_30_n.npz')
# extract the first array

xFake = dict_data['arr_0']

X_M1_One = xFake

yPred_M1_One = model_convlstm.predict(X_M1_One)


classes_M1_One = np.argmax(yPred_M1_One, axis=1)

print(dict_data)

explainer = shap.GradientExplainer(model_convlstm, xFake[:100])
shvalues = explainer.shap_values(xFake[101:111])

# print(len(shvalues))

c0 = shvalues[0]
c1 = shvalues[1]
c2 = shvalues[2]
c3 = shvalues[3]
c4 = shvalues[4]
c5 = shvalues[5]
# print(type(c0))
# print(c0[0,:,0])

print(min(c0[0,:,0]))

print(len(c0[0,:,0]))


plt.plot(c0[0,:,0])
plt.show()

sample = 5
feature = 0
fig, axs = plt.subplots(6,2, dpi=250)
fig.suptitle("Shap values for a sample which it's label is "+str(xFake[sample][0]))

fig.text(0.5, 0.04,'#Sample', ha='center')
fig.text(0.04, 0.5,'Signal Value', va='center', rotation='vertical')


for feature in range(1):

  overallMax = abs(max(list(c0[sample,:,feature])+list(c1[sample,:,feature])+list(c2[sample,:,feature])+list(c3[sample,:,feature])+list(c4[sample,:,feature])+list(c5[sample,:,feature]),key=abs))

  # print(y_test[0])

  for i in range(6):
    reds = []
    blues = []

    for value in shvalues[i][sample,:,feature]:
      if value >= 0:
        reds.append(abs(value)/overallMax)
        if abs(value)/overallMax > 1:
          print(value)
        blues.append(0)
      else:
        reds.append(0)
        blues.append(abs(value)/overallMax)

    y = np.arange(512)
    
    axs[i,feature].plot(y,xFake[sample,:,feature],color = "gray")
    
    # x = c0[0,:,1] #np.array([ v if v < 0 else 0 for v in c0[0,:,1]])
    x = xFake[sample,:,feature]
    y = np.arange(512)

    alphas = np.array(blues)
    rgba_colors = np.zeros((512,4))
    # for red the first column needs to be one
    rgba_colors[:,2] = 1
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas

    axs[i,feature].scatter(y, x, s = 15 ,color=rgba_colors)
    
    
    # -------------------
    # x = c0[0,:,1] #np.array([ v if v >= 0 else 0 for v in c0[0,:,1]])
    x = xFake[sample,:,feature]
    y = np.arange(512)

    alphas = np.array(reds)
    rgba_colors = np.zeros((512,4))
    # for red the first column needs to be one
    rgba_colors[:,0] = 1
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas

    axs[i,feature].scatter(y, x, s = 15 ,color=rgba_colors)
plt.show()
