#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import matplotlib
#matplotlib.use("agg")
import numpy as np
import os
import matplotlib.pyplot as plt	
#from matplotlib.collections import LineCollection
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
#import multiprocessing as mp
#import time
import random
import glob
import tkinter as tk
from tkinter import filedialog as tkd
import re
try:
    import cpickle as pickle
except ImportError:
    import pickle


def present_time():
	from datetime import datetime
	return datetime.now().strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

def random_color(alpha):
    levels = [x/255.0 for x in range(32,256,32)]
    color = [random.choice(levels) for _ in range(3)]
    color.append(alpha)
    return tuple(color)

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)
    


def pickleOpen(fileName):
    with open(fileName, 'rb') as inputFile:
        fileData = pickle.load(inputFile)
    return fileData

def getClusDataForClusID(dirname):
    clusData, winSize, nClus, dirString = pickleOpen(dirname)
    clusters_count = np.array(sorted(clusData, key=len, reverse=True))
    return clusters_count, winSize, nClus, dirString#clusters_count[clusId-1]

baseDir = '/media/aman/New Volume/Aman_thesis/pupaeClusterData/'
# baseDir='/media/fly/data/rawData/PD_pupae_clusterPlots_20201015/PD_pupae_clusterPlots_20201015/'

w1118Dirs = ['20190110_211537_w1118/Plots_20201022_233811/',
             '20190118_192353_w1118/Plots_20201022_234954/',
             '20190125_185115_w1118/Plots_20201023_000256/'
             ]

parkxLRRKDirs = ['20190131_150038_park25xLRRKex1/Plots_20201023_000403/',
                 '20190206_234838_park25xLRRKex1/Plots_20201023_002127/',
                 '20190215_183830_park25xLRRKex1/Plots_20201023_002139/'
                 ]
fname = 'Window_length_100/All_samples_WL=100_per_cluster_nClusters_200.pkl'

w1118ClusIds = [15,55,42]
parkxLRRKClusIds = [30,42,32]

data_w1118_all = [getClusDataForClusID(baseDir+folder+fname) for i, folder in enumerate(w1118Dirs)]
data_parkxLRRK_all = [getClusDataForClusID(baseDir+folder+fname) for i, folder in enumerate(parkxLRRKDirs)]

data_w1118 = [data[0][w1118ClusIds[i]-1] for i, data in enumerate(data_w1118_all)]
data_parkxLRRK = [data[0][parkxLRRKClusIds[i]-1] for i, data in enumerate(data_parkxLRRK_all)]


w1118_mean = np.mean([np.mean(data, axis=0) for data in data_w1118], axis=0)
w1118_std = np.std([np.mean(data, axis=0) for data in data_w1118], axis=0)
park25xLrrk_mean = np.mean([np.mean(data, axis=0) for data in data_parkxLRRK], axis=0)
park25xLrrk_std = np.std([np.mean(data, axis=0) for data in data_parkxLRRK], axis=0)
w1118_sem = w1118_std/np.sqrt(len(w1118Dirs))
park25xLrrk_sem = park25xLrrk_std/np.sqrt(len(parkxLRRKDirs))
winSize = len(data_w1118[0][1])

yLimMin = -2
yLimMax = 10
_ = [plt.plot(np.mean(data, axis=0)) for data in data_w1118]
plt.ylim(yLimMin,yLimMax)
plt.show()
_ = [plt.plot(np.mean(data, axis=0)) for data in data_parkxLRRK]
plt.ylim(yLimMin,yLimMax)
plt.show()

color_1 = 'red'
plt.plot(w1118_mean, color=color_1, alpha=0.5)
plt.ylim(yLimMin,yLimMax)
plt.fill_between(range(winSize), w1118_mean-w1118_std, w1118_mean+w1118_std, color=color_1, alpha=0.2)
color_2 = 'green'
plt.plot(park25xLrrk_mean, color=color_2, alpha=0.5)
plt.ylim(yLimMin,yLimMax)
plt.fill_between(range(winSize), park25xLrrk_mean-park25xLrrk_std, park25xLrrk_mean+park25xLrrk_std, color=color_2, alpha=0.2)
plt.show()


from scipy.spatial import distance
from scipy.stats import ttest_ind as ttest

distances_wp = []
distances_ww = []
distances_pp = []
distances_pw = []
for data in data_w1118:
    dist_w = distance.euclidean(np.mean(data, axis=0), w1118_mean)
    dist_p = distance.euclidean(np.mean(data, axis=0), park25xLrrk_mean)
    distances_ww.append(dist_w)
    distances_wp.append(dist_p)
# plt.plot(distances_ww)
# plt.plot(distances_wp)
# plt.show()
for data in data_parkxLRRK:
    dist_p = distance.euclidean(np.mean(data, axis=0), park25xLrrk_mean)
    dist_w = distance.euclidean(np.mean(data, axis=0), w1118_mean)
    distances_pp.append(dist_p)
    distances_pw.append(dist_w)


def label_pVal(i,j,text,X,Y):
    '''
    https://stackoverflow.com/a/11543637
    '''
    #x = (X[i]+X[j])/2
    y = 1.2*max(Y[i], Y[j])
    #dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':20,'shrinkB':20,'linewidth':2}
    plt.annotate(text, xy=(X[i],y+2), zorder=10)
    plt.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)


def createPlot(pltData1, pltData2, tickLabels, pVal):
    step=0.03
    barStep = 1
    barX = np.arange(1,3,barStep)
    scatterX1 = np.arange(barX[0]-step,barX[0]+step,step)
    scatterX2 = scatterX1+barStep
    err_kwargs = {'zorder':2,'linewidth':2,'ecolor':'k'}
    plt.bar(barX[0], np.mean(pltData1), width=0.4, color='gray')
    plt.bar(barX[1], np.mean(pltData2), width=0.4, color='gray')
    plt.scatter(scatterX1, pltData1, color='r', alpha=0.5, zorder=2)
    plt.errorbar(barX[0], np.mean(pltData1), np.std(pltData1), **err_kwargs)
    plt.errorbar(barX[1], np.mean(pltData2), np.std(pltData1), **err_kwargs)
    plt.scatter(scatterX2, pltData2, color='b', alpha=0.4, zorder=2)
    plt.xlim(0,3)
    plt.ylim(0,10)
    plt.ylabel('Eucledian distance')
    plt.xticks(barX, tickLabels)
    label_pVal(0,1,'p=%0.3f'%(pVal),barX, np.hstack((np.mean(pltData1),np.mean(pltData2))))
    plt.savefig(('---').join(tickLabels)+'.svg')
    plt.savefig(('---').join(tickLabels)+'.pdf')
    plt.show()


pltData =[[distances_ww, distances_wp, ['W_vs_Wmean','W_vs_Pmean']],
           [distances_pp, distances_pw, ['P_vs_Pmean','P_vs_Wmean']],
           [distances_ww, distances_pp, ['W_vs_Wmean','P_vs_Pmean']],
           [distances_wp, distances_pw, ['W_vs_Pmean','P_vs_Wmean']]]

pltDataCurr = pltData[0]
createPlot(pltDataCurr[0], pltDataCurr[1], pltDataCurr[2],ttest(pltDataCurr[0], pltDataCurr[1])[1])
pltDataCurr = pltData[1]
createPlot(pltDataCurr[0], pltDataCurr[1], pltDataCurr[2],ttest(pltDataCurr[0], pltDataCurr[1])[1])
pltDataCurr = pltData[2]
createPlot(pltDataCurr[0], pltDataCurr[1], pltDataCurr[2],ttest(pltDataCurr[0], pltDataCurr[1])[1])
pltDataCurr = pltData[3]
createPlot(pltDataCurr[0], pltDataCurr[1], pltDataCurr[2],ttest(pltDataCurr[0], pltDataCurr[1])[1])

print('distances_ww',distances_ww)
print('distances_wp',distances_wp)
print('distances_pw',distances_pw)
print('distances_pp',distances_pp)

"""
distances_ww [2.018743675519704, 2.401911029267812, 1.487516132404588]
distances_wp [6.390449151463793, 3.1100017664435073, 4.790386992532]
distances_pw [4.5044033009116395, 5.476987140057643, 5.527785982222684]
distances_pp [1.5153260718832884, 2.73139720499551, 3.1122595136779343]
"""
































