# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:47:56 2020

@author: fly
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("agg")
import numpy as np
import os
import matplotlib.pyplot as plt	
from matplotlib.collections import LineCollection
#from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import multiprocessing as mp
import random
import time
import glob
import tkinter as tk
from tkinter import filedialog as tkd
#import Tkinter as tk
#import tkFileDialog as tkd
import re
import itertools
import threading as th
import baseFunctions as bf


try:
    import cPickle as pickle
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

def pickleSave(fileName, obj):
    with open(fileName, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
  

def pickleOpen(fileName):
    with open(fileName, 'rb') as inputFile:
        fileData = pickle.load(inputFile)
    return fileData



def saveClusPlts(clusData, winSize, nClus, dirString):
    '''
    saves all plots based on the data in clusData
    '''
    #clusData, winSize, nClus, dirString = args
    #print clusters_count.shape
    pickleFname = dirString + os.sep + 'All_samples_WL=' + str(winSize) + '_per_cluster_nClusters_' + str(nClus) + '.pkl'
    pickleSave(pickleFname , [clusData, winSize, nClus, dirString])
    
    xLim = winSize
    yLim = 20
    clusters_count = np.array(sorted(clusData, key=len, reverse=True))
    temp_dir = dirString + os.sep + 'All_samples_per_cluster' + '_NC=' + str(nClus)
    os.mkdir(temp_dir)
    for idx, clus in enumerate(clusters_count):
        filename = temp_dir + os.sep + 'WL_' + str(winSize) + '_Cluster_' + str(idx) + '_num_samples_' + str(len(clusters_count[idx])) + '.png'
        clusMean = np.mean(clus, axis=0)
        clusSD = np.std(clus, axis=0)
        color = random_color(alpha=0.6)
        plt.plot(clusMean, color = color, alpha = 0.5)
        plt.fill_between(range(winSize), clusMean-clusSD, clusMean+clusSD, color = color, alpha = 0.2)
        plt.xlim(0, xLim)
        plt.ylim(0, yLim)
        plt.savefig(filename)
        plt.close()
    
    nRows = int(np.sqrt(len(clusters_count)))
    nColumns = int(len(clusters_count)/nRows)+1
    fig, ax =  plt.subplots(nrows=nRows, ncols=nColumns, figsize=(1200, 800))
    clusId = 0
    filename_svg = dirString + os.sep + 'WL_' + str(winSize) + '_All_Twitches.svg'
    filename_png = dirString + os.sep + 'WL_' + str(winSize) + '_All_Twitches.png'
    for i in range(nRows):
        for j in range(nColumns):
            if clusId<len(clusters_count):
                clus = clusters_count[clusId]
                clusMean = np.mean(clus, axis=0)
                clusSD = np.std(clus, axis=0)
                color = random_color(alpha=0.6)
                clusId+=1
                ax[i,j].set_xlim(0, xLim)
                ax[i,j].set_ylim(0, yLim)
                ax[i,j].plot(clusMean, color = color, alpha = 0.8)
                ax[i,j].fill_between(range(winSize), clusMean-clusSD, clusMean+clusSD, color = color, alpha = 0.1)
                ax[i,j].set_title('ClusterId: %d, #Samples %d'%(clusId, len(clus)))
    plt.savefig(filename_svg)
    plt.savefig(filename_png)
    plt.close()
        
#    clusters_count = np.array(sorted(clusData, key=len, reverse=True))
#    temp_dir = dirString + os.sep + 'All_samples_per_cluster' + '_NC=' + str(nClus)
#    os.mkdir(temp_dir)
#    fig, ax = plt.subplots()
#    X = np.arange(winSize)
#    for idx in range(len(clusters_count)):
#        filename = temp_dir + os.sep + 'WL_' + str(winSize) + '_Cluster_' + str(idx) + '_num_samples_' + str(len(clusters_count[idx])) + '.png'
#        pltData = np.asarray(clusters_count[idx])
#        segs = np.zeros((pltData.shape[0], winSize, 2))
#        segs[:, :, 1] = pltData
#        segs[:, :, 0] = X
#        ax.set_xlim(0, winSize)
#        ax.set_ylim(0,50)
#        line_segments = LineCollection(segs, linewidth=0.2, colors=pltColors[:pltData.shape[0]])
#        ax.add_collection(line_segments)
#        ax.set_title('All samples in cluster %i, #Samples = %i' %(idx, pltData.shape[0]))
#        ax.set_xlabel("Time window")
#        ax.set_ylabel("Distance (L2 on pixel differences)")
#        plt.savefig(filename)
#        ax.clear()
#    plt.close()
    
def saveClusPlts_original(clusData, winSize, nClus, dirString):
    '''
    saves all plots based on the data in clusData
    '''
    #clusData, winSize, nClus, dirString = args
    #print clusters_count.shape
    pickleFname = dirString + os.sep + 'All_samples_WL=' + str(winSize) + '_per_cluster_nClusters_' + str(nClus) + '.pkl'
    pickleSave(pickleFname , [clusData, winSize, nClus, dirString])
    xLim = winSize
    yLim = 20
    clusters_count = np.array(sorted(clusData, key=len, reverse=True))
    temp_dir = dirString + os.sep + 'All_samples_per_cluster' + '_NC=' + str(nClus)
    os.mkdir(temp_dir)
    for idx, clus in enumerate(clusters_count):
        filename = temp_dir + os.sep + 'WL_' + str(winSize) + '_Cluster_' + str(idx) + '_num_samples_' + str(len(clusters_count[idx])) + '.png'
        clusMean = np.mean(clus, axis=0)
        clusSD = np.std(clus, axis=0)
        color = random_color(alpha=0.6)
        plt.plot(clusMean, color = color, alpha = 0.5)
        plt.fill_between(range(winSize), clusMean-clusSD, clusMean+clusSD, color = color, alpha = 0.2)
        plt.xlim(0, xLim)
        plt.ylim(0, yLim)
        plt.savefig(filename)
        plt.close()
    
    nRows = int(np.sqrt(len(clusters_count)))
    nColumns = int(len(clusters_count)/nRows)+1
    fig, ax =  plt.subplots(nrows=nRows, ncols=nColumns, figsize=(1200, 800))
    clusId = 0
    filename_svg = dirString + os.sep + 'WL_' + str(winSize) + '_All_Twitches.svg'
    filename_png = dirString + os.sep + 'WL_' + str(winSize) + '_All_Twitches.png'
    for i in range(nRows):
        for j in range(nColumns):
            if clusId<len(clusters_count):
                clus = clusters_count[clusId]
                clusMean = np.mean(clus, axis=0)
                clusSD = np.std(clus, axis=0)
                color = random_color(alpha=0.6)
                clusId+=1
                ax[i,j].set_xlim(0, xLim)
                ax[i,j].set_ylim(0, yLim)
                ax[i,j].plot(clusMean, color = color, alpha = 0.8)
                ax[i,j].fill_between(range(winSize), clusMean-clusSD, clusMean+clusSD, color = color, alpha = 0.1)
                ax[i,j].set_title('ClusterId: %d, #Samples %d'%(clusId, len(clus)))
    plt.savefig(filename_svg)
    plt.savefig(filename_png)
    plt.close()
#    clusters_count = np.array(sorted(clusData, key=len, reverse=True))
#    temp_dir = dirString + os.sep + 'All_samples_per_cluster' + '_NC=' + str(nClus)
#    os.mkdir(temp_dir)
#    fig, ax = plt.subplots()
#    X = np.arange(winSize)
#    for idx in range(len(clusters_count)):
#        filename = temp_dir + os.sep + 'WL_' + str(winSize) + '_Cluster_' + str(idx) + '_num_samples_' + str(len(clusters_count[idx])) + '.png'
#        pltData = np.asarray(clusters_count[idx])
#        segs = np.zeros((pltData.shape[0], winSize, 2))
#        segs[:, :, 1] = pltData
#        segs[:, :, 0] = X
#        ax.set_xlim(0, winSize)
#        ax.set_ylim(0,50)
#        line_segments = LineCollection(segs, linewidth=0.2, colors=pltColors[:pltData.shape[0]])
#        ax.add_collection(line_segments)
#        ax.set_title('All samples in cluster %i, #Samples = %i' %(idx, pltData.shape[0]))
#        ax.set_xlabel("Time window")
#        ax.set_ylabel("Distance (L2 on pixel differences)")
#        plt.savefig(filename)
#        ax.clear()
#    plt.close()
    
def getClusteredData(args):
		# KMeans clustering
		num_clusters, windows, k, dist_vs_num_clusters, dir_string = args
		print ('Started Kmeans labels for %d clusters with %d WL at %s'%(num_clusters, k, present_time()))
		kmeans = KMeans(n_clusters= num_clusters, n_init = 16, max_iter = 256, random_state = 0, verbose=False, n_jobs=32).fit(windows)
		print ('Finished Kmeans labels: %i for %d clusters with %d WL at %s'%(len(kmeans.labels_), num_clusters, k, present_time()))
		clusters_count = [[] for i in range(num_clusters)]
		for i in range(len(kmeans.labels_)):
			clusters_count[kmeans.labels_[i]].append(windows[i])
		clusters_count = np.asarray(clusters_count)
		# Sampling the closest data point to the center of each cluster and plotting
		####labels = np.asarray(kmeans.labels_)
		samples_from_clusters = np.zeros(shape=(num_clusters, k))
		tot_cluster_dist = np.zeros(shape=(num_clusters)) #sum of distance of each point to its cluster
		tot_dist = 0 #sum of total cluster distances for each cluster
		for i in range(len(clusters_count)):
			mn = 1e9
			tot_per_cluster = 0
			for j in range(len(clusters_count[i])):
				x = np.zeros(shape=(k,))
				x = np.asarray(kmeans.cluster_centers_[i])
				dist_1 = np.linalg.norm(x-clusters_count[i][j])
				tot_per_cluster = tot_per_cluster + dist_1
				if(mn > dist_1):
					mn = dist_1
					cluster_sample = clusters_count[i][j]
			samples_from_clusters[i] = cluster_sample
			tot_cluster_dist[i] = tot_per_cluster
			tot_dist = tot_dist + tot_per_cluster
		#store total distance vs number of clusters
		dist_vs_num_clusters.append((np.log(tot_dist), num_clusters))
		#display samples_from_clusters
		print ('Started plotting at %s'%(present_time()))
		fig_name = dir_string + os.sep + 'SamplePerCluster_' + str(num_clusters) + '_clusters_' + str(k) + '_WL.png'
		X = np.arange(k)
		for i in range(len(samples_from_clusters)):
			plt.plot(X, samples_from_clusters[i], linewidth = 0.8)
		plt.xlabel("Time window")
		plt.ylabel("Distance (L2 on pixel differences)")
		plt.title('Sample curve from each cluster - no. of clusters = %i' %num_clusters)
		plt.savefig(fig_name)
		plt.close()
		print('Done with window length ' + str(k) + ' and #clusters = ' + str(num_clusters) + ' at ' + present_time())
#		pltsThread = th.Thread(target=saveClusPlts, args=(clusters_count, k, num_clusters, dir_string))
#		pltsThread.start()
		saveClusPlts(clusters_count, k, num_clusters, dir_string)
		return [clusters_count, k, num_clusters, dir_string]

def plotWindows(args):
	'''
	plot clusters based on the window size
	'''
	global dist
	winSize = args[0]
	nClusters = args[1]

	#preprocessing to store windows	
	k = winSize
	print('Started processing for window length: %i at %s'%(k, present_time()))
	print('len dist = %d'%len(dist))
	windows = np.zeros((len(dist) - k, k))	
	for i in range(len(dist) - k):
		windows[i,:] = dist[i:i+k]
		if i%1000000==0:
			print(i )
	print('===> Processing window length ' + str(k))
	windows = np.asarray(windows)
	dist_vs_num_clusters = []	
	dir_string = dir_string_root + os.sep + 'Window_length_' + str(k)	
	print(dir_string)
	os.mkdir(dir_string)
	print('Started clustering at %s'%(present_time()))
	plotMps = []
	for num_clusters in nClusters:
		plotMps.append(getClusteredData((num_clusters,windows, k, dist_vs_num_clusters, dir_string)))
	#pool = args[2]
#	plotMps = pool.map(getClusteredData, zip(nClusters,
#                                          itertools.repeat(windows),\
#                                          itertools.repeat(k),\
#                                          itertools.repeat(dist_vs_num_clusters),\
#                                          itertools.repeat(dir_string)))
	print ('Done plotting via multiProcessing at %s'%(present_time()))
	#Sum of distances of all clusters vs. number of clusters
	fig_name = dir_string + os.sep + 'sumDist_vs_numClusters_' + str(k) + '.png'
	X = [i[1] for i in dist_vs_num_clusters]
	Y = [i[0] for i in dist_vs_num_clusters]
	plt.plot(X, Y)

	plt.xlabel("No. of clusters")
	plt.ylabel("Total distances (log normalized)")
	plt.title('Sum of distances vs. Number of clusters, Window length = %i' %k)
	plt.savefig(fig_name)
	
	print('Plotting distance vs. number of clusters done for window length of ' + str(k))
	print('\n \n')

def getDist(args):
    return ((30 - args[0])**2 + (30 - args[1])**2)**0.5

def readCsvData(csvFname):
    csvData = []
    for line in open(csvFname).readlines():			
        array = [float(x) for x in line.split(',')]
        csvData.append(array[:2])
    return csvData
nThreads = 24
winSizes = np.array([128], dtype=np.uint16)#np.arange(100,300,25)
nClusters = np.array([1024], dtype=np.uint16)#np.arange(100,200,25)

startTime = time.time()

baseDir='/media/fly/data/rawData/PD_pupae_clusterPlots_20201015/PD_pupae_clusterPlots_20201015/'
baseDir = getFolder(baseDir)
pupaeDirs = bf.getDirList(baseDir)
print("Processing for: ")
for pupaDir in pupaeDirs:
    print(pupaDir)
for pupaDir in pupaeDirs:
    dirName = pupaDir + os.sep + 'csv/'
    dir_string_root = dirName+'../Plots_' + present_time()
    csvFiles = getFiles(dirName, ['*.csv'])
    print ('At %s, Started processing:\n%s'%(present_time(), dirName))
    print ('-----Total number of CsvFiles found: %s-----'%len(csvFiles))
    pool = mp.Pool(processes=nThreads)
    csvData = pool.map(readCsvData, csvFiles)
    pool.close()
    pool.join()
    
    data = []
    for csvD in csvData:
            data.extend(csvD)
    print("Read csv files at %s, now calculating Distances"%present_time())
    pool = mp.Pool(processes=nThreads)
    dist = pool.map(getDist, data)
    pool.close()
    pool.join()
    pltColors = [random_color(alpha=0.6) for _ in range(int(len(data)/5))]
    
    try:
        os.mkdir(dir_string_root)
    except FileExistsError:
        pass
    del data
    
    pool = mp.Pool(processes=nThreads)
    for winsize in winSizes:
        plotWindows([winsize, nClusters, pool])
    pool.close()
    pool.join()
    
    print('Processing finished at: ' + present_time())
    print('Processing finished in: ' + str(round((time.time()-startTime), 2)) + ' seconds')
    








"""
def plotWindows_v1_3(args):
	'''
	plot clusters based on the window size
	'''
	global dist
	winSize = args[0]
	nClusters = args[1]
	#preprocessing to store windows	
	k = winSize
	print('Started processing for window length: %i at %s'%(winSize, present_time()))
	print('len dist = %d'%len(dist))
	windows = np.zeros((len(dist) - k, k))	
	for i in range(len(dist) - k):
		windows[i,:] = dist[i:i+k]
		if i%1000000==0:
			print(i )
	print('===> Processing window length ' + str(k))
	windows = np.asarray(windows)
	dist_vs_num_clusters = []	
	dir_string = dir_string_root + os.sep + 'Window_length_' + str(k)	
	print(dir_string)
	os.mkdir(dir_string)
	#dist_per_cluster = []
	print('Started clustering at %s'%(present_time()))
	plotMps = []
	for num_clusters in nClusters: #range(5,10,5): # number of clusters
		
		# KMeans clustering
		kmeans = KMeans(n_clusters= num_clusters, n_init = 12, max_iter = 300, random_state = 0).fit(windows)

		print ('Kmeans labels: %i at %s'%(len(kmeans.labels_), present_time()))

		clusters_count = [[] for i in range(num_clusters)]
		for i in range(len(kmeans.labels_)):
			clusters_count[kmeans.labels_[i]].append(windows[i])
		clusters_count = np.asarray(clusters_count)

		# Sampling the closest data point to the center of each cluster and plotting
		####labels = np.asarray(kmeans.labels_)
		samples_from_clusters = np.zeros(shape=(num_clusters, k))
		tot_cluster_dist = np.zeros(shape=(num_clusters)) #sum of distance of each point to its cluster
		tot_dist = 0 #sum of total cluster distances for each cluster
		
		for i in range(len(clusters_count)):
			mn = 1e9
			tot_per_cluster = 0
			for j in range(len(clusters_count[i])):
				x = np.zeros(shape=(k,))
				x = np.asarray(kmeans.cluster_centers_[i])
				dist_1 = np.linalg.norm(x-clusters_count[i][j])
				tot_per_cluster = tot_per_cluster + dist_1
				if(mn > dist_1):
					mn = dist_1
					cluster_sample = clusters_count[i][j]
			samples_from_clusters[i] = cluster_sample
			tot_cluster_dist[i] = tot_per_cluster
			tot_dist = tot_dist + tot_per_cluster
		
		#store total distance vs number of clusters
		dist_vs_num_clusters.append((np.log(tot_dist), num_clusters))
	
		#display samples_from_clusters
		fig_name = dir_string + os.sep + 'SamplePerCluster_' + str(num_clusters) + '_clusters_' + str(k) + '_WL.png'
		
		X = np.arange(k)
		for i in range(len(samples_from_clusters)):
			plt.plot(X, samples_from_clusters[i], linewidth = 0.8)

		plt.xlabel("Time window")
		plt.ylabel("Distance (L2 on pixel differences)")
		plt.title('Sample curve from each cluster - no. of clusters = %i' %num_clusters)
		plt.savefig(fig_name)
		#plt.gcf().clear()		
		plt.close()

#		plotMp = mp.Process(target=saveClusPlts, args=(clusters_count, k, num_clusters, dir_string))
#		plotMps.append(plotMp)
#		plotMp.start() 
		plotMps.append([clusters_count, k, num_clusters, dir_string])
		print('Done with window length ' + str(k) + ' and #clusters = ' + str(num_clusters) + ' at ' + present_time())
	for plotMp in plotMps:
		saveClusPlts(*plotMp)
#	print ('Done plotting via multiProcessing with exit code %i at %s'%(plotMp.exitcode, present_time()))
	print ('Done plotting via multiProcessing at %s'%(present_time()))
		
			
	#Sum of distances of all clusters vs. number of clusters
	fig_name = dir_string + os.sep + 'sumDist_vs_numClusters_' + str(k) + '.png'
	X = [i[1] for i in dist_vs_num_clusters]
	Y = [i[0] for i in dist_vs_num_clusters]
	plt.plot(X, Y)

	plt.xlabel("No. of clusters")
	plt.ylabel("Total distances (log normalized)")
	plt.title('Sum of distances vs. Number of clusters, Window length = %i' %k)
	plt.savefig(fig_name)
	
	print('Plotting distance vs. number of clusters done for window length of ' + str(k))
	print('\n \n')
    
"""



#for k in range(5, 100, 5): #length of time window
#	#preprocessing to store windows	
#	windows = []	
#	for i in range(len(data) - k):
#		leg1 = []
#		for j in range(i, i+k):
#			d = ((30 - data[j][0])**2 + (30 - data[j][1])**2)**0.5
#			leg1.append(d)
#		windows.append(leg1)
#	windows = np.asarray(windows)
#	
#	dist_vs_num_clusters = []	
#	dir_string = dir_string_root + '/Window_length_' + str(k)	
#	os.mkdir(dir_string)
#	dist_per_cluster = []
#	
#	for num_clusters in range(5, 100, 5): # number of clusters
#		
#		# KMeans clustering
#		kmeans = KMeans(n_clusters= num_clusters, n_init = 10, max_iter = 300, random_state = 0).fit(windows)
#
#		#print len(kmeans.labels_)
#
#		clusters_count = [[] for i in xrange(num_clusters)]
#		for i in range(len(kmeans.labels_)):
#			clusters_count[kmeans.labels_[i]].append(windows[i])
#		clusters_count = np.asarray(clusters_count)
#
#		# Sampling the closest data point to the center of each cluster and plotting
#		labels = np.asarray(kmeans.labels_)
#		samples_from_clusters = np.zeros(shape=(num_clusters, k))
#		tot_cluster_dist = np.zeros(shape=(num_clusters)) #sum of distance of each point to its cluster
#		tot_dist = 0 #sum of total cluster distances for each cluster
#		
#		for i in range(len(clusters_count)):
#			mn = 1e9
#			tot_per_cluster = 0
#			for j in range(len(clusters_count[i])):
#				x = np.zeros(shape=(k,))
#				x = np.asarray(kmeans.cluster_centers_[i])
#				dist = np.linalg.norm(x-clusters_count[i][j])
#				tot_per_cluster = tot_per_cluster + dist
#				if(mn > dist):
#					mn = dist
#					cluster_sample = clusters_count[i][j]
#			samples_from_clusters[i] = cluster_sample
#			tot_cluster_dist[i] = tot_per_cluster
#			tot_dist = tot_dist + tot_per_cluster
#		
#		#store total distance vs number of clusters
#		dist_vs_num_clusters.append((np.log(tot_dist), num_clusters))
#	
#		#display samples_from_clusters
#		fig_name = dir_string + '/' + 'SamplePerCluster_' + str(num_clusters) + '_clusters_' + str(k) + '_WL.png'
#		
#		X = np.arange(k)
#		for i in range(len(samples_from_clusters)):
#			plt.plot(X, samples_from_clusters[i], linewidth = 0.8)
#
#		plt.xlabel("Time window")
#		plt.ylabel("Distance (L2 on pixel differences)")
#		plt.title('Sample curve from each cluster - no. of clusters = %i' %num_clusters)
#		plt.savefig(fig_name)
#		plt.gcf().clear()		
#
#		#print clusters_count.shape
#		temp_dir = dir_string + '/All_samples_per_cluster' + '_NC=' + str(num_clusters)
#		os.mkdir(temp_dir)
#		for idx in range(len(clusters_count)):
#			X = np.arange(k)
#			filename = temp_dir + '/' + 'Cluster_' + str(idx) + '_num_samples_' + str(len(clusters_count[idx])) + '.png'
#			plt.plot(np.transpose(clusters_count[idx]), linewidth=0.2)
#			plt.ylim(0,50)
#			plt.xlabel("Time window")
#			plt.ylabel("Distance (L2 on pixel differences)")
#			plt.title('All samples in cluster %i' %idx)
#			plt.savefig(filename)
#			#plt.show()
#			plt.gcf().clear()
#		print 'Done with window length ' + str(k) + ' and number of clusters = ' + str(num_clusters) 
#		
#			
#	#Sum of distances of all clusters vs. number of clusters
#	fig_name = dir_string + '/' + 'sumDist_vs_numClusters_' + str(k) + '.png'
#	X = [i[1] for i in dist_vs_num_clusters]
#	Y = [i[0] for i in dist_vs_num_clusters]
#	plt.plot(X, Y)
#
# 	plt.xlabel("No. of clusters")
#	plt.ylabel("Total distances (log normalized)")
#	plt.title('Sum of distances vs. Number of clusters, Window length = %i' %k)
#	plt.savefig(fig_name)
#	
#	print 'Plotting distance vs. number of clusters done for window length of ' + str(k)
#	print '\n \n'

