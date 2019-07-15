#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:51:06 2019

@author: aman
"""

import imageLegTwitchBaseFunctions_tmp_20190713 as imLegTw
import sys
import numpy as np
import os
import glob

try: input = raw_input
except NameError: pass

saveImData = True
saveRoiImage = False

#dirname = '/home/pointgrey/imaging/'
##dirname = '/media/pointgrey/4TB_2/'
dirname = '/media/aman/data/'

#pupaDetails = input('Enter the pupa details : <genotype> - <APF> - <time> - <date>\n')
imResizeFactor = 0.5 #resize image to this factor for display feed of the camera
templateResizeFactor = 4 #resize image to this factor for display of the template
nLegs = 4 # no. of legs to be tracked
nBgs = 2 # no. of background templates to be tracked
trackImSpread = 100  #number of pixles around ROI where the ROI will be tracked

genotype = 'VideoTmp'#pupaDetails.split(' -')[0]
imfolder = 'imageData'
roifolder = 'roi'
csvfolder = 'csv'

try:
    baseDir, imDir, roiDir, csvDir = imLegTw.createDirs(dirname, genotype, imfolder, roifolder, csvfolder)
except:
    print("No directories available, please check!!!")
    sys.exit()

roiVal = [504, 510, 110, 116]
roiColors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
roiBorderThickness = 2

templateKeyDict = {1: 'leg L1', 2: 'leg R1', \
                   3: 'leg L2', 4: 'leg R2', \
                   5: 'Background-1', 6: 'Background-2',\
                   }        

nRois = nLegs+nBgs

logFileName = os.path.join(baseDir, "camloop.txt")
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

vidDir = '/media/aman/data/work/pupaData/20190107_212915_34750TMP/imageData'
vidFlist = glob.glob(os.path.join(vidDir, '*.avi'))

for nVid, vidFname in enumerate(vidFlist):
    roiList = imLegTw.roiSelVid(vidFname, roiList, imArgs, getROI = True)
    dirTime = imLegTw.present_time()
    print(dirTime)
    imLegTw.logFileWrite(logFileName, "Video file : %s"%vidFname, printContent = True)
    trackedValues = imLegTw.decodeNProcParllel(vidFname, roiList, displayfps=100, imArgs = imArgs, nThreads=4)
    np.savetxt(os.path.join(csvDir, dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
    print("\r\nProcesing for video file %d/%d: "%(nVid, len(vidFlist)))


