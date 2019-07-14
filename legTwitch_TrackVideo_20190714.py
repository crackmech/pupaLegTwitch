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
#secondaryDisk = '/media/pointgrey/shared/'
#home = '/home/pointgrey/'
home = '/home/aman/'
dirname = '/media/aman/data/'

vidDir = ''
#pupaDetails = input('Enter the pupa details : <genotype> - <APF> - <time> - <date>\n')
imResizeFactor = 0.5 #resize image to this factor for display feed of the camera
templateResizeFactor = 4 #resize image to this factor for display of the template
nLegs = 4 # no. of legs to be tracked
nBgs = 2 # no. of background templates to be tracked
imDuration = 20      #in minutes
trackImSpread = 30  #number of pixles around ROI where the ROI will be tracked


genotype = ''#pupaDetails.split(' -')[0]
imfolder = 'imageData'
roifolder = 'roi'
csvfolder = 'csv'

try:
    baseDir, imDir, roiDir, csvDir = imLegTw.createDirs(dirname, genotype, imfolder, roifolder, csvfolder)
except:
    print("No directories available, please check!!!")
    sys.exit()

roiVal = [504, 510, 110, 116]
templateList = [np.zeros((10,10), dtype=np.uint8)]

roiColors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
roiBorderThickness = 2

templateKeyDict = {1: 'leg L1', 2: 'leg R1', \
                   3: 'leg L2', 4: 'leg R2', \
                   5: 'Background-1', 6: 'Background-2',\
                   }        

nRois = nLegs+nBgs
roiSelKeys = [ord(str(x)) for x in range(nRois)] # keys to be pressed on keyboard for selecting the roi

logFileName = os.path.join(baseDir, "camloop.txt")
#
#imLegTw.logFileWrite(logFileName, pupaDetails, printContent = False)
roiList = imLegTw.selPreviousRois(roiDir, roiVal, nRois)

SaveDirDuration = int(120/imDuration)
nFrames = int((imDuration*60*100)+1)
nFrames = 601
print(nFrames)

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

vidDir = '/media/aman/data/20190714_050816_12/imageData/'
vidFlist = glob.glob(os.path.join(vidDir, '*.avi'))

for vidFname in vidFlist:
    roiList = imLegTw.roiSelVid(vidFname, roiList, imArgs, getROI = True)
    roiList = imLegTw.displayVid(vidFname, roiList, imArgs)
    
    roiFileName = roiDir+"ROIs_"+imLegTw.present_time()+".txt"
    np.savetxt(roiFileName, roiVal, fmt = '%d', delimiter = ',')
    
    imData = imLegTw.getFrameFromVideo(vidFname, 1)
    print(imData.shape)
    templateList = [imLegTw.getTemplate(imData, roi) for roi in roiList]



#for i, vidFname in enumerate(vidFlist):
#    imLegTw.logFileWrite(logFileName, "Directory : "+saveDir, printContent = True)
#    trackedValues = imLegTw.CapNProc(c, im, roiList, templateList, nFrames, saveDir, saveImData, imArgs)
#    np.savetxt(os.path.join(csvDir, dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
#    print("\r\nProcesing for loop number: "+str(nLoop+1))
#
#c.stop_capture()
#c.disconnect()
#
#imLegTw.logFileWrite(logFileName, '----------------------', printContent = False)



