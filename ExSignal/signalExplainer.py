from cProfile import label
import random
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
import matplotlib.transforms as mtrans
import urllib.request

from keras.models import load_model

import komm
import sys
import glob
import os

from numpy import load


# model_file = 'model_convlstm.h5'
# data_file = 'xFake_30_n.npz'

model_file = 'model_convlstm.h5'
data_file = 'xFake_1730.npz'

model=load_model(model_file)
model.summary()



dict_data = load(data_file)
# extract the first array
print(type(dict_data))
xFake = dict_data['X']
labels = dict_data['y']
channel_names = dict_data['channel_names']
class_names = dict_data['class_names']
print()

# xFake = dict_data['arr_0']

X_M1_One = xFake

yPred_M1_One = model.predict(X_M1_One)


classes_M1_One = np.argmax(yPred_M1_One, axis=1)

numberOfSamples,signalLength,numberOfChannels = xFake.shape

numberOfExplainerFeeds = min(int(numberOfSamples*0.25),450)

explainer = shap.GradientExplainer(model, xFake[:numberOfExplainerFeeds])

numberOfExplainedSamples = 50

shvalues = explainer.shap_values(xFake[numberOfExplainerFeeds+1:numberOfExplainerFeeds+numberOfExplainedSamples+1])
actualLabels = labels[numberOfExplainerFeeds+1:numberOfExplainerFeeds+numberOfExplainedSamples+1]


numberOfClasses = len(shvalues)

sample = random.randint(0,numberOfExplainedSamples-1)
fig, axs = plt.subplots(numberOfClasses,numberOfChannels, dpi=250)

fig.suptitle("Explaining how the model is identifying class "+r"$\bf{" + str(class_names[int(actualLabels[sample][0])]) + "}$"+"?", fontsize=10)

fig.text(0.5, 0.04,'Channels', ha='center', fontsize=10)
fig.text(0.04, 0.5,'Classes', va='center', rotation='vertical', fontsize=10)


for feature in range(numberOfChannels):
    aggList = []
    for class_ in range(numberOfClasses):
        aggList = list(aggList) + list(shvalues[class_][sample,:,feature])
    overallMax = abs(max(aggList,key=abs))
    
    for i in range(numberOfClasses):
        reds = []
        blues = []

        for value in shvalues[i][sample,:,feature]:
            if value >= 0:
                reds.append(abs(value)/overallMax)
                blues.append(0)
            else:
                reds.append(0)
                blues.append(abs(value)/overallMax)

        y = np.arange(signalLength)
        
        if numberOfChannels == 1:
            axs[i].plot(y,xFake[sample,:,feature],color = "gray", lw=0.5, label = "Signal", alpha = 0.5)  
            axs[0].set_title(str(channel_names[feature]), fontsize=8)
            axs[i].set_ylabel(class_names[i], fontsize=6)  
        else:
            axs[i,feature].plot(y,xFake[sample,:,feature],color = "gray", lw=0.5, label = "Signal", alpha = 0.5)
            axs[0,feature].set_title(str(channel_names[feature]), fontsize=8)
            axs[i,0].set_ylabel(class_names[i], fontsize=6)

        x = xFake[sample,:,feature]
        y = np.arange(signalLength)

        alphas = np.array(blues)
        
        rgba_colors = np.zeros((signalLength,4))
        # for red the first column needs to be one
        rgba_colors[:,2] = 1
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = alphas

        if numberOfChannels == 1:
            axs[i].scatter(y, x, s = 15/numberOfChannels ,c=rgba_colors)
        else:
            axs[i,feature].scatter(y, x, s = 15/numberOfChannels ,c=rgba_colors)


        # -------------------
        # x = c0[0,:,1] #np.array([ v if v >= 0 else 0 for v in c0[0,:,1]])
        x = xFake[sample,:,feature]
        y = np.arange(signalLength)

        alphas = np.array(reds)
        rgba_colors = np.zeros((signalLength,4))
        # for red the first column needs to be one
        rgba_colors[:,0] = 1
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = alphas
        
        if numberOfChannels == 1:
            axs[i].scatter(y, x, s = 15/numberOfChannels ,c=rgba_colors)
        else:
            axs[i,feature].scatter(y, x, s = 15/numberOfChannels ,c=rgba_colors)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


#Get the minimum and maximum extent, get the coordinate half-way between those
if numberOfChannels > 1:
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axs.flat)), mtrans.Bbox).reshape(axs.shape)
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axs.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axs.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0.1,.9],[y,y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

plt.show()
