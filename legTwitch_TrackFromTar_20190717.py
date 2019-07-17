#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:51:06 2019

@author: aman
"""
import imageLegTwitchBaseFunctions_tmp_20190713 as imLegTw
import numpy as np
import os
import glob
import multiprocessing as mp

try: input = raw_input
except NameError: pass

saveImData = True
saveRoiImage = False

dirname = '/media/data_ssd/rawMovies'
nThreads = 30

imResizeFactor = 0.5        #resize display image to this factor
templateResizeFactor = 4    #resize template image to this factor for display
nLegs = 4                   # no. of legs to be tracked
nBgs = 2                    # no. of background templates to be tracked
trackImSpread = 30          #number of pixles around ROI where the ROI will be tracked
csvStep = 1000              # stepsize for getting the reference position of the leg

dirname = imLegTw.getFolder(dirname, 'Select Input Directory with videos')
fileExt = '.tar'
genotype = 'VideoTmp'#pupaDetails.split(' -')[0]
imfolder = 'imageData'
roifolder = 'roi'
csvfolder = 'csv'

baseDir, imDir, roiDir, csvDir = imLegTw.createDirsCheck(dirname, genotype, imfolder,  roifolder, csvfolder, 
                                                         baseDir = False, imDir = False, roiDir = True, csvDir = True)

roiVal = [504, 510, 110, 116]
roiColors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
roiBorderThickness = 2

templateKeyDict = {1: 'leg L1', 2: 'leg R1', \
                   3: 'leg L2', 4: 'leg R2', \
                   5: 'Background-1', 6: 'Background-2',\
                   }        

nRois = nLegs+nBgs
logFileName = os.path.join(baseDir, "videoProcessLog.txt")
roiList = imLegTw.selPreviousRois(roiDir, roiVal, nRois)

imLegTw.logFileWrite(logFileName, '----------------------', printContent = False)
imArgs = {'nLegs': nLegs,
          'roiColors': roiColors, 
          'roiBorderThickness': roiBorderThickness, 
          'nRois': nRois,
          'imResizeFactor': imResizeFactor,
          'templateResizeFactor': templateResizeFactor,
          'logfname': logFileName,
          'baseDir': baseDir,
          'imDir': imDir,
          'roiDir': roiDir,
          'csvDir': csvDir,
          'trackImSpread' : trackImSpread,
          'roiVal' : roiVal,
          'templateKeys': templateKeyDict
           }

flist = imLegTw.natural_sort(glob.glob(os.path.join(dirname, imfolder, '*'+fileExt)))
pool = mp.Pool(processes=nThreads)

for nFile, fname in enumerate(flist):
    if nFile == 0:
        roiList = imLegTw.roiSelVid(fname, roiList, imArgs, getROI = True) #select ROIs from the first video file
    dirTime = imLegTw.present_time()
    vidName = fname.split(os.sep)[-1]
    print('Started processing %s (%d/%d) at %s:'%(fname, nFile, len(fname), dirTime))
    imLegTw.logFileWrite(logFileName, "Video file : %s"%fname, printContent = False)
    trackedValues = imLegTw.decodeNProcParllel(fname, roiList, displayfps=100, imArgs = imArgs, pool = pool, nThreads = nThreads)
    np.savetxt(os.path.join(csvDir, vidName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
    eucDisData = imLegTw.csvToData(trackedValues, csvStep, os.path.join(csvDir, vidName+'_'+dirTime+'_XY_eucdisAngles.txt'))
    imLegTw.plotDistance(eucDisData, vidName, os.path.join(csvDir, vidName+'_'+dirTime+'_XY_eucDis.png'))
pool.close()





