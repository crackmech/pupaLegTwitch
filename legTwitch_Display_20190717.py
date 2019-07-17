#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:51:06 2019

@author: aman
"""
import imageLegTwitchBaseFunctions_tmp_20190717_flyPC as imLegTw
import numpy as np
import os
import glob
import multiprocessing as mp

import matplotlib.pyplot as plt



try: input = raw_input
except NameError: pass

saveImData = True
saveRoiImage = False

dirname = '/media/data_ssd/rawMovies'
#dirname = '/media/data_ssd/tmp/CS/tmp (4th copy)/'
nThreads = 30

imResizeFactor = 0.5        #resize display image to this factor
templateResizeFactor = 4    #resize template image to this factor for display
nLegs = 4                   # no. of legs to be tracked
nBgs = 2                    # no. of background templates to be tracked
trackImSpread = 30          #number of pixles around ROI where the ROI will be tracked
csvStep = 1000              # stepsize for getting the reference position of the leg

dirname = imLegTw.getFolder(dirname, 'Select Input Directory with videos')
aviFileExt = '.avi'
tarFileExt = '.tar'
imFileExt = '.jpeg'

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
#roiList = imLegTw.selPreviousRois(roiDir, roiVal, nRois)

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
          'templateKeys': templateKeyDict,
          'nThreads': nThreads
           }


def getTrackedRois(roilist, trackedROI, trackImSpread):
    '''
    returns the ROIs of the given frame based on the tracked values
    '''
    rois = []
    for x, roi in enumerate(roilist):
        delX = (trackedROI[(2*x)]-trackImSpread)
        delY = (trackedROI[(2*x)+1]-trackImSpread)
        currRoi = [roi[0]+delX, roi[1]+delX,
                   roi[2]+delY, roi[3]+delY]
        rois.append(currRoi)
    return rois

def processAvi(imArgs, fileExt, pool, displayTrackedIms = True):
    '''
    process the folder containing the AVI files
    '''
    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "videoProcessLog.txt")
    imLegTw.logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
    roiList = imLegTw.selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
    flist = imLegTw.natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+fileExt)))
    for nFile, fname in enumerate(flist):
        if nFile == 0:
            roiList = imLegTw.roiSelVid(fname, roiList, imArgs, getROI = True) #select ROIs from the first video file
        dirTime = imLegTw.present_time()
        fileName = fname.split(os.sep)[-1]
        print('Started processing VIDEO %s (%d/%d) at %s:'%(fname, nFile, len(flist), dirTime))
        imLegTw.logFileWrite(imArgs['logfname'], "Video file : %s"%fname, printContent = False)
        trackedValues = imLegTw.decodeNProcParllel(fname, roiList, displayfps=100, imArgs = imArgs, pool = pool, nThreads = nThreads)
        if displayTrackedIms:
            i = 0
            cap = cv2.VideoCapture(fname)
            print("Started Display at: %s"%(imLegTw.present_time()))
            ret, imData = cap.read()
            while (ret):
                rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
                imLegTw.displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
                i += 1
                k = cv2.waitKey(100) & 0xFF
                if k == (27):
                    break
            cv2.destroyAllWindows()
        np.savetxt(os.path.join(csvDir, fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
        eucDisData = imLegTw.csvToData(trackedValues, csvStep, os.path.join(csvDir, fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
        imLegTw.plotDistance(eucDisData, fileName, os.path.join(csvDir, fileName+'_'+dirTime+'_XY_eucDis.png'))

import cv2
def processTar(imArgs, fileExt, pool, displayTrackedIms = True):
    '''
    process the folder containing the AVI files
    '''
    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "tarProcessLog.txt")
    imLegTw.logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
    roiList = imLegTw.selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
    flist = imLegTw.natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+fileExt)))
    for nFile, fname in enumerate(flist):
        imNamesDict = imLegTw.tarFolderReadtoDict(tarName = fname, nCurThrds = 0)
        imNamesList = imLegTw.natural_sort(imNamesDict.keys())
        if nFile == 0:
            imData = cv2.imdecode(np.frombuffer(imNamesDict[imNamesList[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
            roiList = imLegTw.roiSelTar(imData, roiList, imArgs, getROI = True)
            templatelist = [imLegTw.getTemplate(imData, roi) for roi in roiList]
        dirTime = imLegTw.present_time()
        fileName = fname.split(os.sep)[-1]
        print('Started processing TAR %s (%d/%d) at %s:'%(fname, nFile, len(flist), dirTime))
        imLegTw.logFileWrite(imArgs['logfname'], "Video file : %s"%fname, printContent = False)
        nFrames = len(imNamesList)
        trackedValues = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
        for i, imKey in enumerate(imNamesList):
            try:
                imData = cv2.imdecode(np.frombuffer(imNamesDict[imKey], np.uint8), cv2.IMREAD_GRAYSCALE)
                trackedValues[i] = imLegTw.trackAllTemplates(templatelist, roiList, imArgs, imData)
                if i == 0:
                    roiFname = os.path.join(roiDir, fileName+'_'+dirTime+'.jpeg')
                    imLegTw.ShowImageWithROI(imData, roiList, roiFname, imArgs, showRoiImage = False, saveRoiImage = True)
            except:
                print(imKey, Exception)
                pass
        if displayTrackedIms:
            for i, imKey in enumerate(imNamesList):
                try:
                    rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
                    imData = cv2.imdecode(np.frombuffer(imNamesDict[imKey], np.uint8), cv2.IMREAD_COLOR)
                    imLegTw.displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
                    k = cv2.waitKey(100) & 0xFF
                    if k == (27):
                        break
                    print (roiList, rois, trackedValues[i])
                except:
                    print(imKey, Exception)
                    break
            cv2.destroyAllWindows()
        #frameStep =  int(nFrames // imArgs['nThreads'])
#        trackedValues = imLegTw.decodeNProcParllel(fname, roiList, displayfps=100, imArgs = imArgs, pool = pool, nThreads = nThreads)
        np.savetxt(os.path.join(csvDir, fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
        eucDisData = imLegTw.csvToData(trackedValues, csvStep, os.path.join(csvDir, fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
        plt.close()
        imLegTw.plotDistance(eucDisData, fileName, os.path.join(csvDir, fileName+'_'+dirTime+'_XY_eucDis.png'))



processFromAvi = True
processFromTar = False
processFromImfolder = False

pool = mp.Pool(processes=nThreads)
if processFromAvi:
    print('Processing from the AVI file')
    processAvi(imArgs, aviFileExt, pool, displayTrackedIms = True)
elif processFromTar:
    print('Processing from the TAR file')
    exception = processTar(imArgs, tarFileExt, pool, displayTrackedIms = True)
elif processFromImfolder:
    print('Processing from the imageData folder')
else:
    print('Select a proper method for processing the video frames')



#imArgs['logfname'] = os.path.join(imArgs['baseDir'], "tarProcessLog.txt")
#imLegTw.logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
#roiList = imLegTw.selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
#flist = imLegTw.natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+tarFileExt)))
#for nFile, fname in enumerate(flist):
#    imNamesDict = imLegTw.tarFolderReadtoDict(tarName = fname, nCurThrds = 0)
#    imNamesList = imLegTw.natural_sort(imNamesDict.keys())
#    if nFile == 0:
#        print(imNamesList[0])
#        imData = cv2.imdecode(np.frombuffer(imNamesDict[imNamesList[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
#        cv2.imshow('tarIM', imData)
#        cv2.waitKey()
#        cv2.destroyAllwindows()


pool.close()



#flist = imLegTw.natural_sort(glob.glob(os.path.join(dirname, imfolder, '*'+fileExt)))
#pool = mp.Pool(processes=nThreads)
#
#for nFile, fname in enumerate(flist):
#    if nFile == 0:
#        roiList = imLegTw.roiSelVid(fname, roiList, imArgs, getROI = True) #select ROIs from the first video file
#    dirTime = imLegTw.present_time()
#    vidName = fname.split(os.sep)[-1]
#    print('Started processing %s (%d/%d) at %s:'%(fname, nFile, len(fname), dirTime))
#    imLegTw.logFileWrite(logFileName, "Video file : %s"%fname, printContent = False)
#    trackedValues = imLegTw.decodeNProcParllel(fname, roiList, displayfps=100, imArgs = imArgs, pool = pool, nThreads = nThreads)
#    np.savetxt(os.path.join(csvDir, vidName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
#    eucDisData = imLegTw.csvToData(trackedValues, csvStep, os.path.join(csvDir, vidName+'_'+dirTime+'_XY_eucdisAngles.txt'))
#    imLegTw.plotDistance(eucDisData, vidName, os.path.join(csvDir, vidName+'_'+dirTime+'_XY_eucDis.png'))
#pool.close()





