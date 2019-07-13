#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 00:19:26 2019

@author: pointgrey
retreive the leg movement distance clustering plots of an animal on per hour basis
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import random
import numpy as np
from datetime import datetime
import glob
import re
import os
import Tkinter as tk
import tkFileDialog as tkd
import multiprocessing as mp
from sklearn.cluster import KMeans


def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def getDirList(folder):
    return natural_sort([os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def getFilesList(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'


def readCsvs(csvName):
    return np.genfromtxt(csvName,dtype = 'float64', delimiter = ',' )


def getFileAPF(pupaCollecDetails, fname):
    '''
    returns the APF hour, the folder name of the corresponding csv/disAngtxt file 
        based on the  csv/disAngtxt file defined by 'fname' 
        if the pupa collection time is pupaCollecDetails
    '''
    fname = fname.split(os.sep)[-1].lstrip('XY_')
    folderName = fname.split('_')[0]+'_'+fname.split('_')[1]
    collDetails = datetime.strptime(pupaCollecDetails, '%Y%m%d_%H%M%S')
    imDetails = datetime.strptime(folderName, '%Y%m%d_%H%M%S')
    apfHour = int(((imDetails-collDetails).total_seconds()) /3600)
    return apfHour, folderName

def getPupaDetails(camFile, DisAngtxtFileList):
    cam = open(camFile, 'r')
    print "Pupa details found"
    for line in cam:
        if 'APF' in line:
            genotype = line.split(' - ')[0]
            collecDetail = line.split(' - ')[3].rstrip('\n')+'_'+line.split(' - ')[2]
    print 'Genotype : ' +genotype+'\n0APF     : '+collecDetail
    startHour, imDetail = getFileAPF(collecDetail, DisAngtxtFileList[0])
    print 'StartHour: '+str(startHour) + 'APF (@'+imDetail+')'
    return imDetail, genotype, startHour

def getAPFList(camFile, DisAngtxtFileList):
    '''
    returns a dictionary of APFs along with the corresponding CSV files of that APF
    '''
    with open(camFile, 'r') as cam: 
        print "Pupa details found"
        for line in cam:
            if 'APF' in line:
                genotype = line.split(' - ')[0]
                collecDetail = line.split(' - ')[3].rstrip('\n')+'_'+line.split(' - ')[2]
    print 'Genotype : ' +genotype+'\n0APF     : '+collecDetail
    apfDict = {}
    for f_, f in enumerate(DisAngtxtFileList):
        apfHour, startHour = getFileAPF(collecDetail, f)
        if apfHour not in apfDict.keys():
            apfDict[apfHour] = {'0APF': collecDetail}
        apfDict[apfHour][str(apfHour)+'_'+str(f_)] = f
    return apfDict

def saveClusPlts(args):
    '''
    saves all plots based on the data in clusData
    '''
    clusData, winSize, nClus, dirString = args
    clusters_count = np.array(sorted(clusData, key=len, reverse=True))
    #print clusters_count.shape
    temp_dir = dirString + os.sep + 'All_samples_per_cluster' + '_NC=' + str(nClus)
    os.mkdir(temp_dir)
    fig, ax = plt.subplots()
    X = np.arange(winSize)
    for idx in range(len(clusters_count)):
        filename = temp_dir + os.sep + 'WL_' + str(winSize) + '_Cluster_' + str(idx) + '_num_samples_' + str(len(clusters_count[idx])) + '.png'
        pltData = np.asarray(clusters_count[idx])
        if len(pltData)>0:
            segs = np.zeros((pltData.shape[0], winSize, 2))
            segs[:, :, 1] = pltData
            segs[:, :, 0] = X
            ax.set_xlim(-1, winSize)
            ax.set_ylim(-1,50)
            line_segments = LineCollection(segs, linewidth=0.2, colors=pltColors[:pltData.shape[0]])
            ax.add_collection(line_segments)
            ax.set_title('All samples in cluster %i, #Samples = %i' %(idx, pltData.shape[0]))
            ax.set_xlabel("Time window")
            ax.set_ylabel("Distance (L2 on pixel differences)")
            plt.savefig(filename)
            ax.clear()
        else:
            ax.text(0.95, 0.01, 'No Clusters found',
                    verticalalignment='center', horizontalalignment='center',
                    transform=ax.transAxes,color='green', fontsize=15)
    plt.close()
    
def random_color(alpha):
    levels = [x/255.0 for x in range(32,256,32)]
    color = [random.choice(levels) for _ in range(3)]
    color.append(alpha)
    return tuple(color)



baseDir = '/home/aman/imaging/w1118'
fileExtension = ['*_XY.csv']
nThreads = 8
k = 40 #sliding window size
num_clusters = 100

apfsToPlot = [87, 100]


apfDataToPlot = {'20190110_211537_w1118': [94,104],
                   '20190118_192353_w1118': [91,102],
                   '20190125_185115_w1118': [91, 102],
                   '20190131_150038_park25xLRRKex1': [87, 101],
                   '20190206_234838_park25xLRRKex1': [84, 99]
                   }
baseDir = getFolder(baseDir)
dirs = getDirList(baseDir)
pltColors = [random_color(alpha=0.6) for _ in xrange(100)]

print "Started on: " + present_time()
apfWindows = [[],[]]
for d in dirs:
    print '-----'+d+'-------'
    currDir = os.path.join(baseDir, d)
    nCsvs = getFilesList(os.path.join(currDir, 'csv'), fileExtension)
    try:
        apfs = getAPFList(os.path.join(currDir, 'camloop.txt'), nCsvs)
    except:
        print "Pupa details unkown"
    apfsToPlot = [apf for apf in apfs.keys() if apf in apfDataToPlot[d.split(os.sep)[-1]]]
    print apfsToPlot
    for ap,apfToPlot in enumerate(apfsToPlot):
        print("processing for %dAPF"%apfToPlot)
        dataFiles = apfs[apfToPlot]
        dataFiles = [apfs[apfToPlot][x] for x in apfs[apfToPlot].keys() if x!='0APF']
        pool = mp.Pool(processes=nThreads)
        data = np.vstack(pool.map(readCsvs, dataFiles))
        pool.close()
    
        disData = np.zeros((len(data)))
        print ("Started calculating EucDis on: %s"%present_time())
        for i in range(len(data)):
            disData[i] = ((30 - data[i][0])**2 + (30 - data[i][1])**2)**0.5
        print ("Started data binning on: %s"%present_time())
        windows = np.zeros((len(data)-k, k))
        for i in range(len(data) - k):
            windows[i] = disData[i:i+k]
        apfWindows[ap].extend(data)


for dataClus in xrange(len(apfWindows)):
    dat = np.vstack(apfWindows[dataClus])
    apfWindows[dataClus] = dat
    print apfWindows[dataClus].shape

for dataClus in xrange(len(apfWindows)):
    print ("Started clustering on: %s"%present_time())
    kmeans = KMeans(n_clusters= num_clusters, n_init = 8, max_iter = 300, random_state = 0, n_jobs=nThreads).fit(windows)
    clusters_count = [[] for i in xrange(num_clusters)]
    print ("---Done clustering on: %s"%present_time())
    for i in range(len(kmeans.labels_)):
        clusters_count[kmeans.labels_[i]].append(windows[i])
    clusters_count = np.asarray(clusters_count)
    print ("Started plotting on: %s"%present_time())
    
    dir_string_root = str(apfToPlot)+'APF_Plots_' + present_time()
    os.mkdir(os.path.join(baseDir, dir_string_root))
    dir_string = os.path.join(baseDir, dir_string_root, 'Window_length_' + str(k))	
    os.mkdir(dir_string)
    
    saveClusPlts([clusters_count, k, num_clusters, dir_string])
    print("---Done plotting on: %s"%present_time())
    

"""
data = []
for line in open("20161120_075105_XY.csv").readlines():			
	array = [float(x) for x in line.split(',')]
	data.append(array)

dir_string_root = 'Plots_' + present_time()
os.mkdir(dir_string_root)

for k in range(5, 100, 5): #length of time window
	#preprocessing to store windows	
	windows = []	
	for i in range(len(data) - k):
		leg1 = []
		for j in range(i, i+k):
			d = ((30 - data[j][0])**2 + (30 - data[j][1])**2)**0.5
			leg1.append(d)
		windows.append(leg1)
	windows = np.asarray(windows)
	
	dist_vs_num_clusters = []	
	dir_string = dir_string_root + '/Window_length_' + str(k)	
	os.mkdir(dir_string)
	dist_per_cluster = []
	
	for num_clusters in range(5, 100, 5): # number of clusters
		
		# KMeans clustering
		kmeans = KMeans(n_clusters= num_clusters, n_init = 10, max_iter = 300, random_state = 0).fit(windows)

		#print len(kmeans.labels_)

		clusters_count = [[] for i in xrange(num_clusters)]
		for i in range(len(kmeans.labels_)):
			clusters_count[kmeans.labels_[i]].append(windows[i])
		clusters_count = np.asarray(clusters_count)

		# Sampling the closest data point to the center of each cluster and plotting
		labels = np.asarray(kmeans.labels_)
		samples_from_clusters = np.zeros(shape=(num_clusters, k))
		tot_cluster_dist = np.zeros(shape=(num_clusters)) #sum of distance of each point to its cluster
		tot_dist = 0 #sum of total cluster distances for each cluster
		
		for i in range(len(clusters_count)):
			mn = 1e9
			tot_per_cluster = 0
			for j in range(len(clusters_count[i])):
				x = np.zeros(shape=(k,))
				x = np.asarray(kmeans.cluster_centers_[i])
				dist = np.linalg.norm(x-clusters_count[i][j])
				tot_per_cluster = tot_per_cluster + dist
				if(mn > dist):
					mn = dist
					cluster_sample = clusters_count[i][j]
			samples_from_clusters[i] = cluster_sample
			tot_cluster_dist[i] = tot_per_cluster
			tot_dist = tot_dist + tot_per_cluster
		
		#store total distance vs number of clusters
		dist_vs_num_clusters.append((np.log(tot_dist), num_clusters))
	
		#display samples_from_clusters
		fig_name = dir_string + '/' + 'SamplePerCluster_' + str(num_clusters) + '_clusters_' + str(k) + '_WL.png'
		
		X = np.arange(k)
		for i in range(len(samples_from_clusters)):
			plt.plot(X, samples_from_clusters[i], linewidth = 0.8)

		plt.xlabel("Time window")
		plt.ylabel("Distance (L2 on pixel differences)")
		plt.title('Sample curve from each cluster - no. of clusters = %i' %num_clusters)
		plt.savefig(fig_name)
		plt.gcf().clear()		

		#print clusters_count.shape
		temp_dir = dir_string + '/All_samples_per_cluster' + '_NC=' + str(num_clusters)
		os.mkdir(temp_dir)
		for idx in range(len(clusters_count)):
			X = np.arange(k)
			filename = temp_dir + '/' + 'Cluster_' + str(idx) + '_num_samples_' + str(len(clusters_count[idx])) + '.png'
			plt.plot(np.transpose(clusters_count[idx]), linewidth=0.2)
			plt.ylim(0,50)
			plt.xlabel("Time window")
			plt.ylabel("Distance (L2 on pixel differences)")
			plt.title('All samples in cluster %i' %idx)
			plt.savefig(filename)
			#plt.show()
			plt.gcf().clear()
		print 'Done with window length ' + str(k) + ' and number of clusters = ' + str(num_clusters) 
		
			
	#Sum of distances of all clusters vs. number of clusters
	fig_name = dir_string + '/' + 'sumDist_vs_numClusters_' + str(k) + '.png'
	X = [i[1] for i in dist_vs_num_clusters]
	Y = [i[0] for i in dist_vs_num_clusters]
	plt.plot(X, Y)

 	plt.xlabel("No. of clusters")
	plt.ylabel("Total distances (log normalized)")
	plt.title('Sum of distances vs. Number of clusters, Window length = %i' %k)
	plt.savefig(fig_name)
	
	print 'Plotting distance vs. number of clusters done for window length of ' + str(k)
	print '\n \n'
	





















"""