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
baseDir='/media/fly/data/rawData/PD_pupae_clusterPlots_20201015/PD_pupae_clusterPlots_20201015/'

fname = 'Window_length_100/All_samples_WL=100_per_cluster_nClusters_200.pkl'

w1118Dirs = ['20190110_211537_w1118/Plots_20201022_233811/',
             '20190118_192353_w1118/Plots_20201022_234954/',
             '20190125_185115_w1118/Plots_20201023_000256/'
             ]

parkxLRRKDirs = ['20190131_150038_park25xLRRKex1/Plots_20201023_000403/',
                 '20190206_234838_park25xLRRKex1/Plots_20201023_002127/',
                 '20190215_183830_park25xLRRKex1/Plots_20201023_002139/'
                 ]

w1118ClusIds = [15,
                55,
                42]
parkxLRRKClusIds = [30,
                    44,
                    29]

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

distances_wp = []
distances_ww = []
distances_pp = []
distances_wp = []
for data in data_w1118:
    dist_w = distance.euclidean(np.mean(data, axis=0), w1118_mean)
    dist_p = distance.euclidean(np.mean(data, axis=0), park25xLrrk_mean)
    distances_ww.append(dist_w)
    distances_wp.append(dist_p)
plt.plot(distances)
plt.show()
print(distances)
for data in data_parkxLRRK:
    dist_w = distance.euclidean(np.mean(data, axis=0), w1118_mean)
    dist_p = distance.euclidean(np.mean(data, axis=0), park25xLrrk_mean)
    distances_ww.append(dist_w)
    distances_wp.append(dist_p)
distances = []
for data in data_parkxLRRK:
    dist_w = distance.euclidean(np.mean(data, axis=0), w1118_mean)
    dist_p = distance.euclidean(np.mean(data, axis=0), park25xLrrk_mean)
    distances.append([dist_w, dist_p])
plt.plot(distances)
plt.show()
print(distances)




from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
u = w1118_mean

v = np.mean(data_w1118[1], axis=0)
u.shape[1] == v.shape[1]
directed_hausdorff(u, v)[0]
w1118_mean.shape
np.mean(data_w1118[1], axis=0).shape

mtx1, mtx2, disparity = procrustes(w1118_mean, np.mean(data_w1118[1], axis=0))
mtx1, mtx2, disparity = procrustes(data_w1118[0][:100], data_parkxLRRK[0][:100])

mtx1, mtx2, disparity = procrustes(data_w1118[0][:100], data_w1118[0][100:200])
mtx1, mtx2, disparity = procrustes(data_parkxLRRK[0][:100], data_parkxLRRK[0][100:200])


#https://lexfridman.com/fast-cross-correlation-and-time-series-synchronization-in-python/
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

# shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift


nClus_w1118 = data_w1118_all[0][2]
nClus_parkxLRRK = data_parkxLRRK_all[0][2]
corrVals_w1118 = np.zeros((nClus_w1118,nClus_w1118), dtype=np.float64)
corrVals_parkxLRRK = np.zeros((nClus_parkxLRRK,nClus_parkxLRRK), dtype=np.float64)


corData_Raw = data_w1118_all[0][0]
corData = [np.mean(clusData, axis=0) for clusData in corData_Raw]

for i, clusData in enumerate(corData):
    for j in range(i, len(corData)):
        corrVals_w1118[i,j] = compute_shift(clusData, corData[j])
        # corrVals_w1118[i,j] = np.argmax(cross_correlation_using_fft(clusData, corData[j]))



    
    
# print(nClus_parkxLRRK)


#
#
#w1118_1 = '20190110_211537_w1118/Plots_20201022_233811/'
#w1118_2 = '20190118_192353_w1118/Plots_20201022_234954/'
#w1118_3 = '20190125_185115_w1118/Plots_20201023_000256/'
#
#park25xLrrk_1 = '20190131_150038_park25xLRRKex1/Plots_20201023_000403/'
#park25xLrrk_2 = '20190206_234838_park25xLRRKex1/Plots_20201023_002127/'
#park25xLrrk_3 = '20190215_183830_park25xLRRKex1/Plots_20201023_002139/'
#
#w1118_1_clus_A_Id = 15
#w1118_2_clus_A_Id = 55
#w1118_3_clus_A_Id = 42
#
#park25xLrrk_1_clus_A_Id = 30
#park25xLrrk_2_clus_A_Id = 44
#park25xLrrk_3_clus_A_Id = 29
#
#data_w1118_1 = pickleOpen(baseDir+w1118_1+fname)
#data_w1118_2 = pickleOpen(baseDir+w1118_2+fname)
#data_w1118_3 = pickleOpen(baseDir+w1118_3+fname)
#
#data_park25xLrrk_1 = pickleOpen(baseDir+park25xLrrk_1+fname)
#data_park25xLrrk_2 = pickleOpen(baseDir+park25xLrrk_2+fname)
#data_park25xLrrk_3 = pickleOpen(baseDir+park25xLrrk_3+fname)
#
#
##clusData, winSize, nClus, dirString = data_w1118_1
#winSize = data_w1118_1[1]
#
#clusters_count_1 = np.array(sorted(data_w1118_1[0], key=len, reverse=True))
#clusters_count_2 = np.array(sorted(data_w1118_2[0], key=len, reverse=True))
#clusters_count_3 = np.array(sorted(data_w1118_3[0], key=len, reverse=True))
#
#clusters_count_4 = np.array(sorted(data_park25xLrrk_1[0], key=len, reverse=True))
#clusters_count_5 = np.array(sorted(data_park25xLrrk_2[0], key=len, reverse=True))
#clusters_count_6 = np.array(sorted(data_park25xLrrk_3[0], key=len, reverse=True))
#
#
#
#yLimMin = -2
#yLimMax = 10
#plt.plot(np.mean(clusters_count_1[w1118_1_clus_A_Id-1], axis=0))
#plt.plot(np.mean(clusters_count_2[w1118_2_clus_A_Id-1], axis=0))
#plt.plot(np.mean(clusters_count_3[w1118_3_clus_A_Id-1], axis=0))
#plt.ylim(yLimMin,yLimMax)
#plt.show()
#plt.plot(np.mean(clusters_count_4[park25xLrrk_1_clus_A_Id-1], axis=0))
#plt.plot(np.mean(clusters_count_5[park25xLrrk_2_clus_A_Id-1], axis=0))
#plt.plot(np.mean(clusters_count_6[park25xLrrk_3_clus_A_Id-1], axis=0))
#plt.ylim(yLimMin,yLimMax)
#plt.show()
#
#w1118_mean = np.mean([np.mean(clusters_count_1[w1118_1_clus_A_Id-1], axis=0),
#                      np.mean(clusters_count_2[w1118_2_clus_A_Id-1], axis=0),
#                      np.mean(clusters_count_3[w1118_3_clus_A_Id-1], axis=0),
#                      ], axis=0)
#w1118_std = np.std([np.mean(clusters_count_1[w1118_1_clus_A_Id-1], axis=0),
#                     np.mean(clusters_count_2[w1118_2_clus_A_Id-1], axis=0),
#                     np.mean(clusters_count_3[w1118_3_clus_A_Id-1], axis=0),
#                      ], axis=0)
#park25xLrrk_mean = np.mean([np.mean(clusters_count_4[park25xLrrk_1_clus_A_Id-1], axis=0),
#                            np.mean(clusters_count_5[park25xLrrk_2_clus_A_Id-1], axis=0),
#                            np.mean(clusters_count_6[park25xLrrk_3_clus_A_Id-1], axis=0),
#                            ], axis=0)
#
#park25xLrrk_std = np.std([np.mean(clusters_count_4[park25xLrrk_1_clus_A_Id-1], axis=0),
#                          np.mean(clusters_count_5[park25xLrrk_2_clus_A_Id-1], axis=0),
#                          np.mean(clusters_count_6[park25xLrrk_3_clus_A_Id-1], axis=0),
#                          ], axis=0)
#nSamplesW1118 = 3
#nSamplesparkxLRRK = 3
#w1118_sem = w1118_std/np.sqrt(nSamplesW1118)
#park25xLrrk_sem = park25xLrrk_std/np.sqrt(nSamplesparkxLRRK)
#
#color_1 = 'red'
#plt.plot(w1118_mean, color=color_1, alpha=0.5)
#plt.ylim(yLimMin,yLimMax)
#plt.fill_between(range(winSize), w1118_mean-w1118_std, w1118_mean+w1118_std, color=color_1, alpha=0.2)
#color_2 = 'green'
#plt.plot(park25xLrrk_mean, color=color_2, alpha=0.5)
#plt.ylim(yLimMin,yLimMax)
#plt.fill_between(range(winSize), park25xLrrk_mean-park25xLrrk_std, park25xLrrk_mean+park25xLrrk_std, color=color_2, alpha=0.2)
#plt.show()
#



#dirName = getFolder('.')
#csvFiles = getFiles(dirName, ['*.csv'])
#print ('-----Total number of CsvFiles found: %s-----'%len(csvFiles))
#data = []
#for csvF in csvFiles[:6]:
#    for line in open(csvF).readlines():			
#        array = [float(x) for x in line.split(',')]
#        data.append(array)
#
#dist = np.zeros((len(data)))
#for i in range(len(data)):
#    dist[i] = ((30 - data[i][0])**2 + (30 - data[i][1])**2)**0.5
#
#
##data = []
###for line in open("20161120_075105_XY.csv").readlines():			
##for line in open("20190107_212925_XY.csv").readlines():			
##	array = [float(x) for x in line.split(',')]
##	data.append(array)
#
#pltColors = [random_color(alpha=0.6) for _ in xrange(len(data)/5)]
#
#dir_string_root = 'Plots_' + present_time()
#os.mkdir(dir_string_root)
#del data
#
#	
#def plotWindows(winSize):
#	'''
#	plot clusters based on the window size
#	'''
#	global dist
#	#preprocessing to store windows	
#	k = winSize
#	print('Started processing for window length: %i at %s'%(winSize, present_time()))
#	windows = np.zeros((len(dist) - k, k))	
#	for i in range(len(dist) - k):
#		windows[i,:] = dist[i:i+k]
#		if i%1000000==0:
#			print i 
#	print '===> Processing window length ' + str(k)
#	windows = np.asarray(windows)
#	dist_vs_num_clusters = []	
#	dir_string = dir_string_root + os.sep + 'Window_length_' + str(k)	
#	print dir_string
#	os.mkdir(dir_string)
#	#dist_per_cluster = []
#	print('Started clustering at %s'%(present_time()))
#	plotMps = []
#	for num_clusters in range(5,100,5): # number of clusters
#		
#		# KMeans clustering
#		kmeans = KMeans(n_clusters= num_clusters, n_init = 12, max_iter = 300, random_state = 0).fit(windows)
#
#		print ('Kmeans labels: %i at %s'%(len(kmeans.labels_), present_time()))
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
#		fig_name = dir_string + os.sep + 'SamplePerCluster_' + str(num_clusters) + '_clusters_' + str(k) + '_WL.png'
#		
#		X = np.arange(k)
#		for i in range(len(samples_from_clusters)):
#			plt.plot(X, samples_from_clusters[i], linewidth = 0.8)
#
#		plt.xlabel("Time window")
#		plt.ylabel("Distance (L2 on pixel differences)")
#		plt.title('Sample curve from each cluster - no. of clusters = %i' %num_clusters)
#		plt.savefig(fig_name)
#		#plt.gcf().clear()		
#		plt.close()
#
##		plotMp = mp.Process(target=saveClusPlts, args=(clusters_count, k, num_clusters, dir_string))
##		plotMps.append(plotMp)
##		plotMp.start() 
#		plotMps.append([clusters_count, k, num_clusters, dir_string])
#		print 'Done with window length ' + str(k) + ' and #clusters = ' + str(num_clusters) + ' at ' + present_time()
#	for plotMp in plotMps:
#		saveClusPlts(*plotMp)
##	print ('Done plotting via multiProcessing with exit code %i at %s'%(plotMp.exitcode, present_time()))
#	print ('Done plotting via multiProcessing at %s'%(present_time()))
#		
#			
#	#Sum of distances of all clusters vs. number of clusters
#	fig_name = dir_string + os.sep + 'sumDist_vs_numClusters_' + str(k) + '.png'
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
#    
#nThreads = 6
#startTime = time.time()
###for x in xrange(100,5,-5):
##for x in xrange(5,100,5):
##    t1 = time.time()
##    plotWindows(x, dist)
##    t2 = time.time()
##    print('for WinLen %i, Time taken: %0.3f seconds'%(x,(t2-t1)))
##    t1 = t2
##
##print('Total time taken: %0.3f seconds'%(time.time()-startTime))
#
#nThreads = 6
#pool = mp.Pool(processes=nThreads)
#winSizes = np.arange(5,100,5)
#_ = pool.map(plotWindows, winSizes)
#
#
#print 'Processing finished at: ' + present_time()
#print 'Processing finished in: ' + str(round((time.time()-startTime), 2)) + ' seconds'
#
#pool.close()
#pool.join()
#
##for k in range(5, 100, 5): #length of time window
##	#preprocessing to store windows	
##	windows = []	
##	for i in range(len(data) - k):
##		leg1 = []
##		for j in range(i, i+k):
##			d = ((30 - data[j][0])**2 + (30 - data[j][1])**2)**0.5
##			leg1.append(d)
##		windows.append(leg1)
##	windows = np.asarray(windows)
##	
##	dist_vs_num_clusters = []	
##	dir_string = dir_string_root + '/Window_length_' + str(k)	
##	os.mkdir(dir_string)
##	dist_per_cluster = []
##	
##	for num_clusters in range(5, 100, 5): # number of clusters
##		
##		# KMeans clustering
##		kmeans = KMeans(n_clusters= num_clusters, n_init = 10, max_iter = 300, random_state = 0).fit(windows)
##
##		#print len(kmeans.labels_)
##
##		clusters_count = [[] for i in xrange(num_clusters)]
##		for i in range(len(kmeans.labels_)):
##			clusters_count[kmeans.labels_[i]].append(windows[i])
##		clusters_count = np.asarray(clusters_count)
##
##		# Sampling the closest data point to the center of each cluster and plotting
##		labels = np.asarray(kmeans.labels_)
##		samples_from_clusters = np.zeros(shape=(num_clusters, k))
##		tot_cluster_dist = np.zeros(shape=(num_clusters)) #sum of distance of each point to its cluster
##		tot_dist = 0 #sum of total cluster distances for each cluster
##		
##		for i in range(len(clusters_count)):
##			mn = 1e9
##			tot_per_cluster = 0
##			for j in range(len(clusters_count[i])):
##				x = np.zeros(shape=(k,))
##				x = np.asarray(kmeans.cluster_centers_[i])
##				dist = np.linalg.norm(x-clusters_count[i][j])
##				tot_per_cluster = tot_per_cluster + dist
##				if(mn > dist):
##					mn = dist
##					cluster_sample = clusters_count[i][j]
##			samples_from_clusters[i] = cluster_sample
##			tot_cluster_dist[i] = tot_per_cluster
##			tot_dist = tot_dist + tot_per_cluster
##		
##		#store total distance vs number of clusters
##		dist_vs_num_clusters.append((np.log(tot_dist), num_clusters))
##	
##		#display samples_from_clusters
##		fig_name = dir_string + '/' + 'SamplePerCluster_' + str(num_clusters) + '_clusters_' + str(k) + '_WL.png'
##		
##		X = np.arange(k)
##		for i in range(len(samples_from_clusters)):
##			plt.plot(X, samples_from_clusters[i], linewidth = 0.8)
##
##		plt.xlabel("Time window")
##		plt.ylabel("Distance (L2 on pixel differences)")
##		plt.title('Sample curve from each cluster - no. of clusters = %i' %num_clusters)
##		plt.savefig(fig_name)
##		plt.gcf().clear()		
##
##		#print clusters_count.shape
##		temp_dir = dir_string + '/All_samples_per_cluster' + '_NC=' + str(num_clusters)
##		os.mkdir(temp_dir)
##		for idx in range(len(clusters_count)):
##			X = np.arange(k)
##			filename = temp_dir + '/' + 'Cluster_' + str(idx) + '_num_samples_' + str(len(clusters_count[idx])) + '.png'
##			plt.plot(np.transpose(clusters_count[idx]), linewidth=0.2)
##			plt.ylim(0,50)
##			plt.xlabel("Time window")
##			plt.ylabel("Distance (L2 on pixel differences)")
##			plt.title('All samples in cluster %i' %idx)
##			plt.savefig(filename)
##			#plt.show()
##			plt.gcf().clear()
##		print 'Done with window length ' + str(k) + ' and number of clusters = ' + str(num_clusters) 
##		
##			
##	#Sum of distances of all clusters vs. number of clusters
##	fig_name = dir_string + '/' + 'sumDist_vs_numClusters_' + str(k) + '.png'
##	X = [i[1] for i in dist_vs_num_clusters]
##	Y = [i[0] for i in dist_vs_num_clusters]
##	plt.plot(X, Y)
##
## 	plt.xlabel("No. of clusters")
##	plt.ylabel("Total distances (log normalized)")
##	plt.title('Sum of distances vs. Number of clusters, Window length = %i' %k)
##	plt.savefig(fig_name)
##	
##	print 'Plotting distance vs. number of clusters done for window length of ' + str(k)
##	print '\n \n'
#
