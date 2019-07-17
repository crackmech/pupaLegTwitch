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

<<<<<<< HEAD

=======
>>>>>>> 2250ed9cf05c37c82af81e40c1ca39c2d4b61dd6
try: input = raw_input
except NameError: pass

saveImData = True
saveRoiImage = False

<<<<<<< HEAD
#dirname = '/home/pointgrey/imaging/'
##dirname = '/media/pointgrey/4TB_2/'
<<<<<<< HEAD
#secondaryDisk = '/media/pointgrey/shared/'
#home = '/home/pointgrey/'
home = '/home/aman/'
dirname = '/media/aman/data/'

vidDir = ''
=======
dirname = '/media/aman/data/'

>>>>>>> 2250ed9cf05c37c82af81e40c1ca39c2d4b61dd6
#pupaDetails = input('Enter the pupa details : <genotype> - <APF> - <time> - <date>\n')
imResizeFactor = 0.5 #resize image to this factor for display feed of the camera
templateResizeFactor = 4 #resize image to this factor for display of the template
nLegs = 4 # no. of legs to be tracked
nBgs = 2 # no. of background templates to be tracked
<<<<<<< HEAD
imDuration = 20      #in minutes
trackImSpread = 30  #number of pixles around ROI where the ROI will be tracked


genotype = ''#pupaDetails.split(' -')[0]
=======
trackImSpread = 100  #number of pixles around ROI where the ROI will be tracked
=======
dirname = '/media/data_ssd/rawMovies'
nThreads = 30

imResizeFactor = 0.5        #resize display image to this factor
templateResizeFactor = 4    #resize template image to this factor for display
nLegs = 4                   # no. of legs to be tracked
nBgs = 2                    # no. of background templates to be tracked
trackImSpread = 30          #number of pixles around ROI where the ROI will be tracked
csvStep = 1000              # stepsize for getting the reference position of the leg
>>>>>>> e4989ef88ec6acf025349009697604b564dd7f30

dirname = imLegTw.getFolder(dirname, 'Select Input Directory with videos')
genotype = 'VideoTmp'#pupaDetails.split(' -')[0]
>>>>>>> 2250ed9cf05c37c82af81e40c1ca39c2d4b61dd6
imfolder = 'imageData'
roifolder = 'roi'
csvfolder = 'csv'

baseDir, imDir, roiDir, csvDir = imLegTw.createDirsCheck(dirname, genotype, imfolder,  roifolder, csvfolder, 
                                                         baseDir = False, imDir = False, roiDir = True, csvDir = True)

roiVal = [504, 510, 110, 116]
<<<<<<< HEAD
templateList = [np.zeros((10,10), dtype=np.uint8)]

=======
>>>>>>> 2250ed9cf05c37c82af81e40c1ca39c2d4b61dd6
roiColors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
roiBorderThickness = 2

templateKeyDict = {1: 'leg L1', 2: 'leg R1', \
                   3: 'leg L2', 4: 'leg R2', \
                   5: 'Background-1', 6: 'Background-2',\
                   }        

nRois = nLegs+nBgs
<<<<<<< HEAD
<<<<<<< HEAD
roiSelKeys = [ord(str(x)) for x in range(nRois)] # keys to be pressed on keyboard for selecting the roi

logFileName = os.path.join(baseDir, "camloop.txt")
#
#imLegTw.logFileWrite(logFileName, pupaDetails, printContent = False)
roiList = imLegTw.selPreviousRois(roiDir, roiVal, nRois)

SaveDirDuration = int(120/imDuration)
nFrames = int((imDuration*60*100)+1)
nFrames = 601
print(nFrames)

=======

logFileName = os.path.join(baseDir, "camloop.txt")
=======
logFileName = os.path.join(baseDir, "videoProcessLog.txt")
>>>>>>> e4989ef88ec6acf025349009697604b564dd7f30
roiList = imLegTw.selPreviousRois(roiDir, roiVal, nRois)

>>>>>>> 2250ed9cf05c37c82af81e40c1ca39c2d4b61dd6
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

<<<<<<< HEAD
<<<<<<< HEAD
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

=======
vidDir = '/media/aman/data/work/pupaData/20190107_212915_34750TMP/imageData'
vidFlist = glob.glob(os.path.join(vidDir, '*.avi'))
=======
vidFlist = imLegTw.natural_sort(glob.glob(os.path.join(dirname, imfolder, '*.avi')))
pool = mp.Pool(processes=nThreads)
>>>>>>> e4989ef88ec6acf025349009697604b564dd7f30

for nVid, vidFname in enumerate(vidFlist):
    if nVid == 0:
        roiList = imLegTw.roiSelVid(vidFname, roiList, imArgs, getROI = True) #select ROIs from the first video file
    dirTime = imLegTw.present_time()
<<<<<<< HEAD
    print(dirTime)
    imLegTw.logFileWrite(logFileName, "Video file : %s"%vidFname, printContent = True)
    trackedValues = imLegTw.decodeNProcParllel(vidFname, roiList, displayfps=100, imArgs = imArgs, nThreads=4)
    np.savetxt(os.path.join(csvDir, dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
    print("\r\nProcesing for video file %d/%d: "%(nVid, len(vidFlist)))
>>>>>>> 2250ed9cf05c37c82af81e40c1ca39c2d4b61dd6
=======
    vidName = vidFname.split(os.sep)[-1]
    print('Started processing %s (%d/%d) at %s:'%(vidFname, nVid, len(vidFlist), dirTime))
    imLegTw.logFileWrite(logFileName, "Video file : %s"%vidFname, printContent = False)
    trackedValues = imLegTw.decodeNProcParllel(vidFname, roiList, displayfps=100, imArgs = imArgs, pool = pool, nThreads = nThreads)
    np.savetxt(os.path.join(csvDir, vidName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
    eucDisData = imLegTw.csvToData(trackedValues, csvStep, os.path.join(csvDir, vidName+'_'+dirTime+'_XY_eucdisAngles.txt'))
    imLegTw.plotDistance(eucDisData, vidName, os.path.join(csvDir, vidName+'_'+dirTime+'_XY_eucDis.png'))
pool.close()



>>>>>>> e4989ef88ec6acf025349009697604b564dd7f30


