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
import cv2
import itertools
#import matplotlib.pyplot as plt

#def getTrackedRois(roilist, trackedROI, trackImSpread):
#    '''
#    returns the ROIs of the given frame based on the tracked values
#    '''
#    rois = []
#    for x, roi in enumerate(roilist):
#        delX = (trackedROI[(2*x)]-trackImSpread)
#        delY = (trackedROI[(2*x)+1]-trackImSpread)
#        currRoi = [roi[0]+delX, roi[1]+delX,
#                   roi[2]+delY, roi[3]+delY]
#        rois.append(currRoi)
#    return rois
#
#def processAvi(imArgs, fileExt, pool, displayTrackedIms = True):
#    '''
#    process the folder containing the AVI files
#    '''
#    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "videoProcessLog.txt")
#    imLegTw.logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
#    roiList = imLegTw.selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
#    flist = imLegTw.natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+fileExt)))
#    for nFile, fname in enumerate(flist):
#        if nFile == 0:
#            roiList = imLegTw.roiSelVid(fname, roiList, imArgs, getROI = True) #select ROIs from the first video file
#        dirTime = imLegTw.present_time()
#        fileName = fname.split(os.sep)[-1]
#        print('Started processing VIDEO %s (%d/%d) at %s:'%(fname, nFile+1, len(flist), dirTime))
#        imLegTw.logFileWrite(imArgs['logfname'], "Video file : %s"%fname, printContent = False)
#        trackedValues = imLegTw.decodeNProcParllel(fname, roiList, displayfps=100, imArgs = imArgs, pool = pool, nThreads = imArgs['nThreads'])
#        if displayTrackedIms:
#            i = 0
#            cap = cv2.VideoCapture(fname)
#            print("Started Display at: %s"%(imLegTw.present_time()))
#            ret, imData = cap.read()
#            while (ret):
#                rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
#                imLegTw.displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
#                i += 1
#                if i%1000 == 0:
#                    print('Displayed %d images at %s'%(i, imLegTw.present_time()))
#                k = cv2.waitKey(100) & 0xFF
#                if k == (27):
#                    break
#            cv2.destroyAllWindows()
#            cap.release()
#        np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
#        eucDisData = imLegTw.csvToData(trackedValues, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
#        imLegTw.plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))
#
#def processTar(imArgs, fileExt, pool, displayTrackedIms = True):
#    '''
#    process the folder containing the AVI files
#    '''
#    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "tarProcessLog.txt")
#    imLegTw.logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
#    roiList = imLegTw.selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
#    flist = imLegTw.natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+fileExt)))
#    for nFile, fname in enumerate(flist):
#        imNamesDict = imLegTw.tarFolderReadtoDict(tarName = fname, nCurThrds = 0)
#        imNamesList = imLegTw.natural_sort(imNamesDict.keys())
#        if nFile == 0:
#            imData = cv2.imdecode(np.frombuffer(imNamesDict[imNamesList[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
#            roiList = imLegTw.roiSelTar(imData, roiList, imArgs, getROI = True)
#            templatelist = [imLegTw.getTemplate(imData, roi) for roi in roiList]
#        dirTime = imLegTw.present_time()
#        fileName = fname.split(os.sep)[-1]
#        print('Started processing TAR %s (%d/%d) at %s:'%(fname, nFile+1, len(flist), dirTime))
#        imLegTw.logFileWrite(imArgs['logfname'], "Tar file : %s"%fname, printContent = False)
#        nFrames = len(imNamesList)
#        trackedValues = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
#        for i, imKey in enumerate(imNamesList):
#            try:
#                imData = cv2.imdecode(np.frombuffer(imNamesDict[imKey], np.uint8), cv2.IMREAD_GRAYSCALE)
#                trackedValues[i] = imLegTw.trackAllTemplates(templatelist, roiList, imArgs, imData)
#                if i == 0:
#                    roiFname = os.path.join(imArgs['roiDir'], fileName+'_'+dirTime+'.jpeg')
#                    imLegTw.ShowImageWithROI(imData, roiList, roiFname, imArgs, showRoiImage = False, saveRoiImage = True)
#            except:
#                print(imKey, Exception)
#                pass
#        if displayTrackedIms:
#            for i, imKey in enumerate(imNamesList):
#                try:
#                    rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
#                    imData = cv2.imdecode(np.frombuffer(imNamesDict[imKey], np.uint8), cv2.IMREAD_COLOR)
#                    imLegTw.displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
#                    k = cv2.waitKey(100) & 0xFF
#                    if k == (27):
#                        break
#                    print (roiList, rois, trackedValues[i])
#                except:
#                    print(imKey, Exception)
#                    break
#            cv2.destroyAllWindows()
#        np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
#        eucDisData = imLegTw.csvToData(trackedValues, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
#        imLegTw.plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))
#
#
#
#def processImFolders(imArgs, fileExt, pool, displayTrackedIms = True):
#    '''
#    process the folder containing the AVI files
#    '''
#    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "ImFolderProcessLog.txt")
#    imLegTw.logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
#    roiList = imLegTw.selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
#    flist = imLegTw.natural_sort(imLegTw.getDirList(imArgs['imDir']))
#    for nFile, fname in enumerate(flist):
#        imNamesList = imLegTw.natural_sort(glob.glob(os.path.join(fname, '*'+fileExt)))
#        if nFile == 0:
#            imData = cv2.imread(imNamesList[nFile], cv2.IMREAD_GRAYSCALE)
#            roiList = imLegTw.roiSelTar(imData, roiList, imArgs, getROI = True)
#            templatelist = [imLegTw.getTemplate(imData, roi) for roi in roiList]
#        dirTime = imLegTw.present_time()
#        fileName = fname.split(os.sep)[-1]
#        print('Started processing FOLDER %s (%d/%d) at %s:'%(fname, nFile+1, len(flist), dirTime))
#        imLegTw.logFileWrite(imArgs['logfname'], "Image folder: %s"%fname, printContent = False)
#        nFrames = len(imNamesList)
#        trackedValues = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
#        for i, imKey in enumerate(imNamesList):
#            try:
#                imData = cv2.imread(imKey, cv2.IMREAD_GRAYSCALE)
#                trackedValues[i] = imLegTw.trackAllTemplates(templatelist, roiList, imArgs, imData)
#                if i == 0:
#                    roiFname = os.path.join(imArgs['roiDir'], fileName+'_'+dirTime+'.jpeg')
#                    imLegTw.ShowImageWithROI(imData, roiList, roiFname, imArgs, showRoiImage = False, saveRoiImage = True)
#                if i%1000 == 0:
#                    print('Processed %d images at %s'%(i, imLegTw.present_time()))
#            except:
#                print(imKey, Exception)
#                pass
#        if displayTrackedIms:
#            for i, imKey in enumerate(imNamesList):
#                try:
#                    rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
#                    imData = cv2.imread(imKey, cv2.IMREAD_COLOR)
#                    imLegTw.displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
#                    k = cv2.waitKey(100) & 0xFF
#                    if k == (27):
#                        break
#                    print (roiList, rois, trackedValues[i])
#                except:
#                    print(imKey, Exception)
#                    break
#            cv2.destroyAllWindows()
#        np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
#        eucDisData = imLegTw.csvToData(trackedValues, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
#        imLegTw.plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))
#
#def procTarIms(poolArgs):
#    '''
#    returns the coordinates of tracked ROIs in the given image list. 
#        imNamesList contains keys from the imNamesDict
#    '''
#    imBuffStack, roilist, templatelist, imArgs, threadN = poolArgs
#    print('Started processing for thread: %d'%threadN)
#    legCoords = np.zeros((len(imBuffStack),2*imArgs['nRois']), dtype = np.uint16)
#    for i, im in enumerate(imBuffStack):
#        try:
#            imData = cv2.imdecode(np.frombuffer(im, np.uint8), cv2.IMREAD_GRAYSCALE)
#            legCoords[i] = imLegTw.trackAllTemplates(templatelist, roilist, imArgs, imData)
#        except:
#            print(Exception)
#            pass
#    return legCoords
#
#def processTarParallel(imArgs, fileExt, pool, displayTrackedIms = True):
#    '''
#    process the folder containing the Tar files
#    '''
#    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "tarProcessLog.txt")
#    imLegTw.logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
#    roiList = imLegTw.selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
#    flist = imLegTw.natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+fileExt)))
#    for nFile, fname in enumerate(flist):
#        imNamesDict = imLegTw.tarFolderReadtoDict(tarName = fname, nCurThrds = 0)
#        imNamesList = imLegTw.natural_sort(imNamesDict.keys())
#        if nFile == 0:
#            imData = cv2.imdecode(np.frombuffer(imNamesDict[imNamesList[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
#            roiList = imLegTw.roiSelTar(imData, roiList, imArgs, getROI = True)
#            templatelist = [imLegTw.getTemplate(imData, roi) for roi in roiList]
#        dirTime = imLegTw.present_time()
#        fileName = fname.split(os.sep)[-1]
#        print('Started processing TAR file %s (%d/%d) at %s:'%(fname, nFile+1, len(flist), dirTime))
#        imLegTw.logFileWrite(imArgs['logfname'], "Tar file: %s"%fname, printContent = False)
#        nFrames = len(imNamesList)
#        trackedValues = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
#        frameStep =  int(nFrames // imArgs['nThreads'])
#        framesList = [imNamesList[i*frameStep:(i+1)*frameStep] for i in range(imArgs['nThreads'])]
#        imBuffStack = [[imNamesDict[y] for y in x] for x in framesList]
#        imData = cv2.imdecode(np.frombuffer(imNamesDict[imNamesList[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
#        roiFname = os.path.join(imArgs['roiDir'], fileName+'_'+dirTime+'.jpeg')
#        imLegTw.ShowImageWithROI(imData, roiList, roiFname, imArgs, showRoiImage = False, saveRoiImage = True)
#        mpArgs = zip(imBuffStack, 
#                     itertools.repeat(roiList), \
#                     itertools.repeat(templatelist), \
#                     itertools.repeat(imArgs), \
#                     range(imArgs['nThreads'])
#                     )
#        print('Done with generating arguments for parallel processing at: %s'%imLegTw.present_time())
#        legCoords = pool.map(procTarIms, mpArgs)
#        trackedValues = np.zeros((len(legCoords)*legCoords[0].shape[0], legCoords[0].shape[1]), dtype=np.uint16)
#        for i, coords in enumerate(legCoords):
#            trackedValues[i*frameStep:(i+1)*frameStep] = coords
#        if displayTrackedIms:
#            for i, imKey in enumerate(imNamesList):
#                try:
#                    rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
#                    imData = cv2.imdecode(np.frombuffer(imNamesDict[imKey], np.uint8), cv2.IMREAD_COLOR)
#                    imLegTw.displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
#                    k = cv2.waitKey(100) & 0xFF
#                    if k == (27):
#                        break
#                    print (roiList, rois, trackedValues[i])
#                except:
#                    print(imKey, Exception)
#                    break
#            cv2.destroyAllWindows()
#        np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
#        eucDisData = imLegTw.csvToData(trackedValues, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
#        imLegTw.plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))
#
dirname = '/media/data_ssd/tmp/CS/tmp_120k'
nThreads = 30

imResizeFactor = 0.5        #resize display image to this factor
templateResizeFactor = 4    #resize template image to this factor for display
nLegs = 4                   # no. of legs to be tracked
nBgs = 2                    # no. of background templates to be tracked
trackImSpread = 30          #number of pixles around ROI where the ROI will be tracked
csvStep = 1000              # stepsize for getting the reference position of the leg

aviFileExt = '.avi'
tarFileExt = '.tar'
imFileExt = '.jpeg'

genotype = 'VideoTmp'#pupaDetails.split(' -')[0]
imfolder = 'imageData'
roifolder = 'roi'
csvfolder = 'csv'

roiVal = [504, 510, 110, 116]
roiColors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
roiBorderThickness = 2

templateKeyDict = {1: 'leg L1', 2: 'leg R1', \
                   3: 'leg L2', 4: 'leg R2', \
                   5: 'Background-1', 6: 'Background-2',\
                   }        

nRois = nLegs+nBgs
#roiList = imLegTw.selPreviousRois(roiDir, roiVal, nRois)
dirname = imLegTw.getFolder(dirname, 'Select Input Directory with videos')
baseDir, imDir, roiDir, csvDir = imLegTw.createDirsCheck(dirname, genotype, imfolder,  roifolder, csvfolder, 
                                                         baseDir = False, imDir = False, roiDir = True, csvDir = True)
logFileName = os.path.join(baseDir, "videoProcessLog.txt")
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
          'csvStep': csvStep, 
          'roiVal' : roiVal,
          'templateKeys': templateKeyDict,
          'nThreads': nThreads
           }




processFromAvi = False
processFromTar = True
processFromImfolder = False

pool = mp.Pool(processes=nThreads)
if processFromAvi:
    print('Processing from the AVI file')
    imLegTw.processAvi(imArgs, aviFileExt, pool, displayTrackedIms = False)
elif processFromTar:
    print('Processing from the TAR file')
#    imLegTw.processTar(imArgs, tarFileExt, pool, displayTrackedIms = False)
    imLegTw.processTarParallel(imArgs, tarFileExt, pool, displayTrackedIms = False)
elif processFromImfolder:
    print('Processing from the imageData folder')
    imLegTw.processImFolders(imArgs, imFileExt, pool, displayTrackedIms = False)
else:
    print('Select a proper method for processing the video frames')


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





