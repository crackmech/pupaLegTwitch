# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:25:35 2019

@author: fly
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:58:22 2016

@author: pointgrey
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:05:21 2015

@author: pointgrey

v1.1: 
    New:
        1) Select Background ROI also for processing
        2) Doesn't convert csv into distAngle txt, used to cause crashes
        3) Variable saveImDir control the image saving process
Script for selecting five different regions of interest for tracking in pupa.
First four ROIs are from the different legs (L1, R1, L2 and R2), and the fifth
ROI is from the background. The output is 

"""
import sys
if sys.version_info[0] < 3:
    import tkFileDialog as tkd
    import Tkinter as tk
else:
    import tkinter.filedialog as tkd
    import tkinter as tk

#import flycapture2 as fc2
import numpy as np
import cv2
from datetime import datetime
import sys
import threading as th
import os
import itertools
import glob
import re
import tarfile
import time

#from thread import start_new_thread as startNT
#import tkFileDialog as tkd
#import Tkinter as tk
#import multiprocessing as mp
#import subprocess as sp
#import math
#import matplotlib.pyplot as plt


try: input = raw_input
except NameError: pass

def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def getFolder(initialDir, title):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title=title)
    root.destroy()
    return initialDir+'/'

def getDirList(folder):
    return natural_sort([os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def logFileWrite(logfname, content, printContent = False):
    '''Create log files'''
    try:
        if printContent:
            print(content)
        with open(logfname, "a") as f:
            f.write('\n'+content)
    except:
        print('Something went wrong! Can\'t create log file')

def getStartSeg():
    '''
    gets the segment value for starting the imaging loop from this value.
    This can be used to start the imaging loop if imaging is interupted due
    to any reason
    '''
    try:
        startSegment = sys.argv[1]
    except:
        startSegment = input('Enter starting segment value : ')
    return int(startSegment)


def createDirs(dirname, genotype, imfolder, roiFolder, csvFolder):
    '''
    creates directories for saving images and csvs
    '''
    #create base directory for all data using current date as the foldername
    try:
        baseDir = os.path.join(dirname, present_time()+'_'+genotype)
        os.mkdir(baseDir)
        #create directory for saving captured images in the base directory
        imDir= os.path.join(baseDir, imfolder)
        #create directory for saving ROI images and ROI files in the base directory
        roiDir = os.path.join(baseDir, roiFolder)
        csvDir = os.path.join(baseDir, csvFolder)
        os.mkdir(imDir)
        os.mkdir(roiDir)
        os.mkdir(csvDir)
        return baseDir, imDir, roiDir, csvDir
    except:
        print("Not able to create directories")
        pass

def createDir(dirname):
    '''
    '''
    try:
        os.mkdir(dirname)
    except:
        print('Not able to create directory: %s'%dirname)

def createDirsCheck(dirname, genotype, imfolder, roiFolder, csvFolder, 
                    baseDir = True, imDir = True, roiDir = True, csvDir = True):
    '''
    creates directories for saving images and csvs
    '''
    #create base directory for all data using current date as the foldername
    dirs = os.listdir(dirname)
    
    dirList = {'baseDir' : dirname, 'imDir' : '', 'roiDir' : '', 'csvDir' : ''}
    try:
        tmpFname = os.path.join(dirname, 'tmpFolder')
        os.mkdir(tmpFname)
        os.rmdir(tmpFname)
        if baseDir:
            baseDir = os.path.join(dirname, present_time()+'_'+genotype)
            if baseDir not in dirs:
                createDir(baseDir)
                dirname = baseDir
        imDir= os.path.join(dirname, imfolder)
        roiDir = os.path.join(dirname, roiFolder)
        csvDir = os.path.join(dirname, csvFolder)
        if imDir:
            if imDir not in dirs:
                createDir(imDir)
        if roiDir:
            if roiDir not in dirs:
                createDir(roiDir)
        if csvDir:
            if csvDir not in dirs:
                createDir(csvDir)
        dirList['baseDir'] = dirname
        dirList['imDir'] = imDir
        dirList['roiDir'] = roiDir
        dirList['csvDir'] = csvDir
        return dirList['baseDir'], dirList['imDir'], dirList['roiDir'], dirList['csvDir']
    except:
        print("Not able to create output directories , please check!!!\nExiting the code")
        sys.exit()

def selPreviousRois(roiDir, roival, nRois):
    '''
    returns the ROIs by reading the ROI values from a previous ROI file
    '''
    try:
        tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename = tkd.askopenfilename(initialdir = roiDir, filetypes = [("Text files","*.txt")]) # show an "Open" dialog box and return the path to the selected file
        print("Using ROIs from :"+filename)
        rois = np.genfromtxt(filename, dtype = np.uint16, delimiter = ',')
        roilist = [list(x) for x in rois]
    except:
        print("Not using previously selected ROIs, select new ROIs")
        roilist = [roival for x in range(nRois)]
    return roilist

def resizeImage(imData, resizefactor):
    '''
    resizes the image to half of the original dimensions
    '''
    newx,newy = int(imData.shape[1]*resizefactor),int(imData.shape[0]*resizefactor)
    resizedImg = cv2.resize(imData,(newx,newy))
    return resizedImg

drawing = False # true if mouse is pressed
ix,iy = -1,-1
img2 = []
def DrawROI(event,x,y,flags,param):
    '''
    function for drawing ROI on the image.
    '''
    global img, img2, ix,iy,drawing, rect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img=img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
        rect = (ix,x,iy,y)

def selRoi(imData, roilist, imArgs, \
           getROI = True, showRoiImage = True, saveRoiImage = True):
    '''
    Function for selecting the ROI on the image. Key press '1', '2', '3' and '4'
    saves the corresponding ROIs in as ROI for 'L1', 'R1', 'L2', 'L2' legs of 
    the pupa. Pressing '5' selects for the background.
    Boolean values dictate if you want to do what is described in the variable
    '''
    global img, img2, rect
    imresizefactor = imArgs['imResizeFactor']
    templateresizefactor = imArgs['templateResizeFactor']
    roiSelKeys = [ord(str(x+1)) for x in range(len(roilist))] 
    try:
        imDataColor = cv2.cvtColor(imData,cv2.COLOR_GRAY2BGR)
    except:
        imDataColor = imData
    if getROI == True:
        rect = ()
        img = resizeImage(imDataColor, imresizefactor)
        img2 = img.copy()
        cv2.namedWindow("Select ROI, Press 'Esc' to exit")
        cv2.setMouseCallback("Select ROI, Press 'Esc' to exit", DrawROI)
        while(1):
            cv2.imshow("Select ROI, Press 'Esc' to exit", img)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                if imArgs['roiVal'] in roilist:
                    pressKey = roilist.index(imArgs['roiVal'])+1
                    print('Select ROI for %s before exiting selection window (by pressing %d)'%(imArgs['templateKeys'][pressKey], pressKey))
                else:
                    break
            if k in roiSelKeys:
                legId = imArgs['templateKeys'][int(chr(k))]
                roi = [int(1/imresizefactor)*x for x in rect]
                roilist[int(chr(k))-1] = roi
                template = getTemplate(imDataColor, roi)
                winName = 'Selected template for %s'%legId
                cv2.namedWindow(winName)
                cv2.imshow(winName, resizeImage(template, templateresizefactor))
                cv2.waitKey()
                cv2.destroyWindow(winName)
                print(winName)
        roiVal = np.asarray(roilist)
        cv2.destroyAllWindows()
        roiFileName = os.path.join(imArgs['roiDir'], "ROIs_"+present_time()+".txt")
        np.savetxt(roiFileName, roiVal, fmt = '%d', delimiter = ',')
        logFileWrite(imArgs['logfname'], "Selected new ROIs", printContent = False)
    roiImageName = os.path.join(imArgs['roiDir'], present_time()+"_ROI.jpeg")
    roilist = ShowImageWithROI(imDataColor, roilist, roiImageName, imArgs, showRoiImage, saveRoiImage)
    return roilist

def ShowImageWithROI(imData, roilist, roifname, imArgs, showRoiImage = True, saveRoiImage = False):
    '''
    Displays the imData with ROIs that are selected.
    If saveRoiImage is true, the image with ROIs marked on it is saved
    '''
    imgData = imData.copy()
    imresizeFactor = imArgs['imResizeFactor']
    try:
        for i, roi in enumerate(roilist):
            if i >= imArgs['nLegs']:
                i = imArgs['nLegs']
            cv2.rectangle(imData, (roi[0], roi[2]), (roi[1], roi[3]), 
                          imArgs['roiColors'][i], imArgs['roiBorderThickness'])
        img = resizeImage(imData, imresizeFactor)
        
        if saveRoiImage == True:
            roiIm = cv2.cvtColor(imgData, cv2.COLOR_GRAY2BGR)
            for i, roi in enumerate(roilist):
                if i >= imArgs['nLegs']:
                    i = imArgs['nLegs']
                cv2.rectangle(roiIm, (roi[0], roi[2]), (roi[1], roi[3]), 
                              imArgs['roiColors'][i], imArgs['roiBorderThickness'])
            cv2.imwrite(roifname, roiIm)
            logFileWrite(imArgs['logfname'], 'Saved ROI Image as : '+ roifname, printContent = False)
            
        if showRoiImage == True:
            while(1):
                cv2.imshow("Selected ROIs, Press 'Esc' to exit, press'r' to reselect ROIs", img)
                k = cv2.waitKey(10) & 0xFF
                if k == 27:
                    break
                if k == ord("r"):
                    print("Selecting ROIs again")
                    cv2.destroyAllWindows()
                    roilist = selRoi(imgData, roilist, imArgs, \
                                     getROI = True, showRoiImage = True, saveRoiImage = False)
                    break
        cv2.destroyAllWindows()
    except:
        print('No Folder / ROIs selected')
    return roilist

def getTemplate(imData, roi):
    '''
    Returns the template from the given image data using roi values
    '''
    template = imData[roi[2]:roi[3], roi[0]:roi[1]]
    if len(imData.shape)>2:
        if imData.shape[-1]<3:
            return template
        else:
            return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        return template

def trackTemplate(template, roi, imData, trackImSpread):
    '''
    Returns the XY values of the template after tracking in the given number of
    frames. Tracking is done using cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    function from opnCV library. 'template' is tracked in the 'imData' file.
    'template' is an array of values of selected ROI in grayscale. 
    'roi' is a list of X,Y locations and dimensions of the roi selected.
    'trackImSpread' is the number of pixels around roi where the template is tracked.
    '''
    img = imData
    img1 = img[roi[2]-trackImSpread:roi[3]+trackImSpread, \
                    roi[0]-trackImSpread:roi[1]+trackImSpread]
    result = cv2.matchTemplate(img1,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc

def trackAllTemplates(templatelist, roilist, imArgs, imData):
    '''
    '''
    trackCoords = np.zeros(2*len(templatelist), dtype=np.uint16)
    for i, template in enumerate(templatelist):
        coords = trackTemplate(template, roilist[i], imData, imArgs['trackImSpread'])
        trackCoords[2*i:(2*i)+2] = coords
    #print(i, coords, trackCoords)
    return trackCoords
    
def displayImageWithROI(windowName, imData, rois, imArgs):
    '''
    Continously display the images from camera with ROIs that are selected
    '''
    for i, roi in enumerate(rois):
        if i >= imArgs['nLegs']:
            i = imArgs['nLegs']
        cv2.rectangle(imData, (roi[0], roi[2]), (roi[1], roi[3]), imArgs['roiColors'][i], imArgs['roiBorderThickness'])
    img = resizeImage(imData, imArgs['imResizeFactor'])
    cv2.imshow(windowName, img)

def displayIm(windowName, imData, sleepTime):
    '''
    '''
    cv2.imshow(windowName, imData)
    return cv2.waitKey(sleepTime) & 0xFF


#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
def initiateCam():
    '''
    initiates the first camera attached to the computer and returns the pointer for the image
    '''
    import flycapture2 as fc2
    #Initiate camera
    c = fc2.Context()
    c.connect(*c.get_camera_from_index(0))
    print("roiSel : %s"%present_time())
    p = c.get_property(fc2.FRAME_RATE)
    print(p)
    c.set_property(**p)
    c.start_capture()
    im = fc2.Image()
    return c, im
    

def roiSelCam(roilist, imArgs, getROI = True):
    '''
    Standalone function to select ROIs from an image captured by the camera
    '''
    #Initiate camera
    c, im = initiateCam()
    #Capture image
    c.retrieve_buffer(im)
    imData = np.array(im)
    #Select ROI from the captured image
    roilist = selRoi(imData, roilist, imArgs, \
                   getROI, showRoiImage = True, saveRoiImage = False)
    c.stop_capture()
    c.disconnect()
    return roilist

def displayCam(c, im, roilist, imArgs):
    '''
    Starts display from the connected camera. ROIs can be updated by pressing 'u'
    '''
    while(1):
        c.retrieve_buffer(im)
        imData = np.array(im)
        imDataColor = cv2.cvtColor(imData,cv2.COLOR_GRAY2BGR)
        windowName = "Camera Display, Press 'u' to update ROIs or 'Esc' to close"
        displayImageWithROI(windowName, imDataColor, roilist, imArgs)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord("u"):
            roilist = selRoi(imData, roilist, imArgs, \
                              getROI = True, showRoiImage = True, saveRoiImage = True)
    cv2.destroyWindow(windowName)
    return roilist


def CapNProc(c, im, roilist, templatelist, nFrames, saveDir, saveIm, imArgs):
    '''
    Function for capturing images from the camera and then tracking already defined
    templates. The template images are updated (using ROIs defined earlier)
    everytime the function is called. Pressing 'Ctrl+c' pauses the tracking loop
    and starts displaying live images from the camera. This can be used to select
    new templates while the function is running.
    '''
    logfname = imArgs['logfname']
    print("%s Press 'Ctrl+C' to pause analysis and start live display"%present_time())
    legCoords = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
    logFileWrite(logfname, present_time(), printContent = False)
    for nFrame in range (0,nFrames):
        try:
            if nFrame%100 == 0:
                sys.stdout.write("\r%s: %d"%(present_time(),nFrame))
                sys.stdout.flush()
            c.retrieve_buffer(im)
            imData = np.array(im)
            if nFrame == 10:
                roilist = selRoi(imData, roilist, imArgs, \
                       getROI = False, showRoiImage = False, saveRoiImage = True)
                templatelist = [getTemplate(imData, roi) for roi in roilist]
                if np.median(imData)>250: # to stop imaging if the image goes white predominantely
                    #print("\n-------No pupa to Image anymore!!-------")
                    logFileWrite(logfname, "------- No pupa to image anymore!! Imaging exited -------", printContent = True)
                    sys.exit(0)
            legCoords[nFrame] = trackAllTemplates(templatelist, roilist, imArgs, imData)
            #for i, template in enumerate(templatelist):
            #    legCoords[nFrame, i:i+2] = trackTemplate(template, roilist[i], imData, imArgs['trackImSpread'])
            #print('\ni: %d\n nFrame: %d\ntemplateList length: %d\n roiList length: %d'%(i,nFrame, len(templatelist), len(roilist)))
            if saveIm == True:
                try:
                    th.Thread(target = cv2.imwrite, args = (os.path.join(saveDir, str(nFrame)+'.jpeg'),imData,))
                except:
                    print("error saving %s"+str(nFrame))
            elif saveIm == False:
                if nFrame%1000 == 0:
                    th.Thread(target = cv2.imwrite, args = (os.path.join(saveDir, str(nFrame)+'.jpeg'),imData,))
        except KeyboardInterrupt:
            logFileWrite(logfname, "\nCamera display started on %s"%present_time(), printContent = True)
            roilist = displayCam(c, im, roilist, imArgs)
            logFileWrite(logfname, "Camera display exited on %s"%present_time(), printContent = False)
    logFileWrite(logfname, present_time(), printContent = False)
    logFileWrite(logfname, '----------------------', printContent = False)
    values = legCoords#np.c_[l1,r1,l2,r2,bg,bg1]
    return values


#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
def roiSelVid(vfname, roilist, imArgs, getROI = True):
    '''
    Standalone function to select ROIs from the video
    '''
    cap = cv2.VideoCapture(vfname)
    while(cap.isOpened()):
        ret, frame = cap.read()
        imData = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(10)
        roilist = selRoi(imData, roilist, imArgs, \
                         getROI, showRoiImage = True, saveRoiImage = False)
        break
    cap.release()
    return roilist

def getFrameFromVideo(vfname, frameN):
    '''
    returns the specified frame number from the video file
    '''
    cap = cv2.VideoCapture(vfname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameN-1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print('No frame at given position')

def displayVid(vfname, roilist, imArgs, fps):
    '''
    Starts displaying Video from the input video file,
    update the ROIs by pressing 'u'
    '''
    cap = cv2.VideoCapture(vfname)
    while(cap.isOpened()):
        ret, imData = cap.read()
        if ret:
            displayImageWithROI('Press "u" to update ROIs', imData, roilist, imArgs)
            k = cv2.waitKey(int(1000/fps)) & 0xFF
            if k == ord('q'):
                break
            if k == ord("u"):
                roilist = selRoi(imData, roilist, imArgs, \
                                  getROI = True, showRoiImage = True, saveRoiImage = True)
        else:
            cap.release()
            cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    return roilist

def procVideo(poolArgs):
    vfname, roilist, templatelist, frameStep, imArgs, groupNumber = poolArgs
    cap = cv2.VideoCapture(vfname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameStep * groupNumber)
    legCoords = np.zeros((frameStep,2*imArgs['nRois']), dtype = np.uint16)
    proc_frames = 0
    while proc_frames < frameStep:
        ret, imData = cap.read()
        if ret:
            imData = cv2.cvtColor(imData, cv2.COLOR_RGB2GRAY)
        legCoords[proc_frames] = trackAllTemplates(templatelist, roilist, imArgs, imData)
        proc_frames += 1
    cap.release()
    return legCoords


def decodeNProcParllel(vfname, roilist, displayfps, imArgs, pool, nThreads):
    '''
    returns the tracking data for the selected ROIs from the video file
    '''
    logfname = imArgs['logfname']
    logFileWrite(logfname, present_time(), printContent = False)
    startTime = time.time()
    cap = cv2.VideoCapture(vfname)
    nFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    imData = getFrameFromVideo(vfname, frameN=0)
    imData = cv2.cvtColor(imData, cv2.COLOR_RGB2GRAY)
    templatelist = [getTemplate(imData, roi) for roi in roilist]
    frameStep =  int(nFrames / imArgs['nThreads'])
    print("Started processing with frameStep %d, on %d threads at: %s"%(frameStep, nThreads, present_time()))
    mpArgs = zip(itertools.repeat(vfname), 
                 itertools.repeat(roilist), \
                 itertools.repeat(templatelist), \
                 itertools.repeat(frameStep), \
                 itertools.repeat(imArgs), \
                 range(nThreads)
                 )
    legCoordsStack = pool.map(procVideo, mpArgs)
    roiFname = os.path.join(imArgs['roiDir'], vfname.split(os.sep)[-1]+'_'+present_time()+'.jpeg')
    ShowImageWithROI(imData, roilist, roiFname, imArgs, showRoiImage = False, saveRoiImage = True)
    timeTaken = time.time()-startTime
    print('Done processing in %0.2f Seconds, at: %s (%0.3fFPS)' %(timeTaken, present_time(), nFrames/timeTaken))
    return np.vstack((legCoordsStack))


#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
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


#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
def processAvi(imArgs, fileExt, pool, displayTrackedIms = True):
    '''
    process the folder containing the AVI files
    '''
    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "videoProcessLog.txt")
    logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
    roiList = selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
    flist = natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+fileExt)))
    for nFile, fname in enumerate(flist):
        if nFile == 0:
            roiList = roiSelVid(fname, roiList, imArgs, getROI = True) #select ROIs from the first video file
        dirTime = present_time()
        fileName = fname.split(os.sep)[-1]
        print('Started processing VIDEO %s (%d/%d) at %s:'%(fname, nFile+1, len(flist), dirTime))
        logFileWrite(imArgs['logfname'], "Video file : %s"%fname, printContent = False)
        trackedValues = decodeNProcParllel(fname, roiList, displayfps=100, imArgs = imArgs, pool = pool, nThreads = imArgs['nThreads'])
        if displayTrackedIms:
            i = 0
            cap = cv2.VideoCapture(fname)
            print("Started Display at: %s"%(present_time()))
            ret, imData = cap.read()
            while (ret):
                rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
                displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
                i += 1
                if i%1000 == 0:
                    print('Displayed %d images at %s'%(i, present_time()))
                k = cv2.waitKey(100) & 0xFF
                if k == (27):
                    break
            cv2.destroyAllWindows()
            cap.release()
        np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
        eucDisData = csvToData(trackedValues, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
        plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))



#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
def processImFolders(imArgs, fileExt, pool, displayTrackedIms = True):
    '''
    process the folder containing the AVI files
    '''
    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "ImFolderProcessLog.txt")
    logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
    roiList = selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
    flist = natural_sort(getDirList(imArgs['imDir']))
    for nFile, fname in enumerate(flist):
        imNamesList = natural_sort(glob.glob(os.path.join(fname, '*'+fileExt)))
        if nFile == 0:
            imData = cv2.imread(imNamesList[nFile], cv2.IMREAD_GRAYSCALE)
            roiList = selRoi(imData, roiList, imArgs,getROI = True, 
                             showRoiImage = True, saveRoiImage = False)
            templatelist = [getTemplate(imData, roi) for roi in roiList]
        dirTime = present_time()
        fileName = fname.split(os.sep)[-1]
        print('Started processing FOLDER %s (%d/%d) at %s:'%(fname, nFile+1, len(flist), dirTime))
        logFileWrite(imArgs['logfname'], "Image folder: %s"%fname, printContent = False)
        nFrames = len(imNamesList)
        trackedValues = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
        for i, imKey in enumerate(imNamesList):
            try:
                imData = cv2.imread(imKey, cv2.IMREAD_GRAYSCALE)
                trackedValues[i] = trackAllTemplates(templatelist, roiList, imArgs, imData)
                if i == 0:
                    roiFname = os.path.join(imArgs['roiDir'], fileName+'_'+dirTime+'.jpeg')
                    ShowImageWithROI(imData, roiList, roiFname, imArgs, showRoiImage = False, saveRoiImage = True)
                if i%1000 == 0:
                    print('Processed %d images at %s'%(i, present_time()))
            except:
                print(imKey, Exception)
                pass
        if displayTrackedIms:
            for i, imKey in enumerate(imNamesList):
                try:
                    rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
                    imData = cv2.imread(imKey, cv2.IMREAD_COLOR)
                    displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
                    k = cv2.waitKey(100) & 0xFF
                    if k == (27):
                        break
                    print (roiList, rois, trackedValues[i])
                except:
                    print(imKey, Exception)
                    break
            cv2.destroyAllWindows()
        np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
        eucDisData = csvToData(trackedValues, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
        plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))



#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
def tarFolderReadtoDict(tarName, nCurThrds):
    '''
    read contents of the imageData tar folder into a dict
    '''
    print('Reading tar file from %s # Current Threads: %d '%(tarName, nCurThrds))
    readTime = time.time()
    tar = tarfile.open(tarName,'r|') 
    tarStack = {}
    for f in tar:
        if f.isfile():
            c = tar.extractfile(f).read()
            fname = f.get_info()['name']
            tarStack[fname] = c
    tar.close()
    print('Read %s at %s in: %.02f Seconds, # Current Threads: %d '%(\
            tarName, present_time(), (time.time()-readTime), nCurThrds))
    return tarStack

def procTarIms(poolArgs):
    '''
    returns the coordinates of tracked ROIs in the given image list. 
        imNamesList contains keys from the imNamesDict
    '''
    imBuffStack, roilist, templatelist, imArgs, threadN = poolArgs
    startTime = time.time()
    print('Started processing at %s for thread: %d'%(present_time(), threadN))
    legCoords = np.zeros((len(imBuffStack),2*imArgs['nRois']), dtype = np.uint16)
    for i, im in enumerate(imBuffStack):
        try:
            imData = cv2.imdecode(np.frombuffer(im, np.uint8), cv2.IMREAD_GRAYSCALE)
            legCoords[i] = trackAllTemplates(templatelist, roilist, imArgs, imData)
        except:
            print(Exception)
            pass
    print('Completed processing at %s for thread: %d (%0.2fFPS)'%(present_time(), threadN, len(imBuffStack)/(time.time()-startTime)))
    return legCoords

def processTarParallel(imArgs, fileExt, pool, displayTrackedIms = True):
    '''
    process the folder containing the Tar files
    '''
    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "tarProcessLog.txt")
    logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
    roiList = selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
    flist = natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+fileExt)))
    for nFile, fname in enumerate(flist):
        imNamesDict = tarFolderReadtoDict(tarName = fname, nCurThrds = 0)
        imNamesList = natural_sort(imNamesDict.keys())
        if nFile == 0:
            imData = cv2.imdecode(np.frombuffer(imNamesDict[imNamesList[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
            roiList = selRoi(imData, roiList, imArgs,getROI = True, 
                             showRoiImage = True, saveRoiImage = False)
            templatelist = [getTemplate(imData, roi) for roi in roiList]
        dirTime = present_time()
        fileName = fname.split(os.sep)[-1]
        print('Started processing TAR file %s (%d/%d) at %s:'%(fname, nFile+1, len(flist), dirTime))
        logFileWrite(imArgs['logfname'], "Tar file: %s"%fname, printContent = False)
        nFrames = len(imNamesList)
        trackedValues = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
        frameStep =  int(nFrames // imArgs['nThreads'])
        framesList = [imNamesList[i*frameStep:(i+1)*frameStep] for i in range(imArgs['nThreads'])]
        imBuffStack = [[imNamesDict[y] for y in x] for x in framesList]
        imData = cv2.imdecode(np.frombuffer(imNamesDict[imNamesList[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
        roiFname = os.path.join(imArgs['roiDir'], fileName+'_'+dirTime+'.jpeg')
        ShowImageWithROI(imData, roiList, roiFname, imArgs, showRoiImage = False, saveRoiImage = True)
        mpArgs = zip(imBuffStack, 
                     itertools.repeat(roiList), \
                     itertools.repeat(templatelist), \
                     itertools.repeat(imArgs), \
                     range(imArgs['nThreads'])
                     )
        print('Done with generating arguments for parallel processing at: %s'%present_time())
        legCoords = pool.map(procTarIms, mpArgs)
        trackedValues = np.zeros((len(legCoords)*legCoords[0].shape[0], legCoords[0].shape[1]), dtype=np.uint16)
        for i, coords in enumerate(legCoords):
            trackedValues[i*frameStep:(i+1)*frameStep] = coords
        if displayTrackedIms:
            for i, imKey in enumerate(imNamesList):
                try:
                    rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
                    imData = cv2.imdecode(np.frombuffer(imNamesDict[imKey], np.uint8), cv2.IMREAD_COLOR)
                    displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
                    k = cv2.waitKey(100) & 0xFF
                    if k == (27):
                        break
                    print (roiList, rois, trackedValues[i])
                except:
                    print(imKey, Exception)
                    break
            cv2.destroyAllWindows()
        np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
        eucDisData = csvToData(trackedValues, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
        plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))


#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
def processTar(imArgs, fileExt, pool, displayTrackedIms = True):
    '''
    process the folder containing the AVI files
    '''
    imArgs['logfname'] = os.path.join(imArgs['baseDir'], "tarProcessLog.txt")
    logFileWrite(imArgs['logfname'], '----------------------', printContent = True)
    roiList = selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
    flist = natural_sort(glob.glob(os.path.join(imArgs['imDir'], '*'+fileExt)))
    for nFile, fname in enumerate(flist):
        imNamesDict = tarFolderReadtoDict(tarName = fname, nCurThrds = 0)
        imNamesList = natural_sort(imNamesDict.keys())
        if nFile == 0:
            imData = cv2.imdecode(np.frombuffer(imNamesDict[imNamesList[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
            roiList = selRoi(imData, roiList, imArgs,getROI = True, 
                             showRoiImage = True, saveRoiImage = False)
            templatelist = [getTemplate(imData, roi) for roi in roiList]
        dirTime = present_time()
        fileName = fname.split(os.sep)[-1]
        print('Started processing TAR %s (%d/%d) at %s:'%(fname, nFile+1, len(flist), dirTime))
        logFileWrite(imArgs['logfname'], "Tar file : %s"%fname, printContent = False)
        nFrames = len(imNamesList)
        trackedValues = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
        for i, imKey in enumerate(imNamesList):
            try:
                imData = cv2.imdecode(np.frombuffer(imNamesDict[imKey], np.uint8), cv2.IMREAD_GRAYSCALE)
                trackedValues[i] = trackAllTemplates(templatelist, roiList, imArgs, imData)
                if i == 0:
                    roiFname = os.path.join(imArgs['roiDir'], fileName+'_'+dirTime+'.jpeg')
                    ShowImageWithROI(imData, roiList, roiFname, imArgs, showRoiImage = False, saveRoiImage = True)
            except:
                print(imKey, Exception)
                pass
        if displayTrackedIms:
            for i, imKey in enumerate(imNamesList):
                try:
                    rois = getTrackedRois(roiList, trackedValues[i], imArgs['trackImSpread'])
                    imData = cv2.imdecode(np.frombuffer(imNamesDict[imKey], np.uint8), cv2.IMREAD_COLOR)
                    displayImageWithROI('Displaying Tracked Legs', imData, rois, imArgs)
                    k = cv2.waitKey(100) & 0xFF
                    if k == (27):
                        break
                    print (roiList, rois, trackedValues[i])
                except:
                    print(imKey, Exception)
                    break
            cv2.destroyAllWindows()
        np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
        eucDisData = csvToData(trackedValues, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
        plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))

def decodeNProc(vfname, roilist, displayfps, imArgs):
    '''
    returns the tracking data for the selected ROIs from the video file
    '''
    logfname = imArgs['logfname']
    logFileWrite(logfname, present_time(), printContent = False)
    print("%s Press 'p' to pause analysis and start live display"%present_time())
    cap = cv2.VideoCapture(vfname)
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    legCoords = np.zeros((nFrames,2*imArgs['nRois']), dtype = np.uint16)
    nFrame = 0
    while(cap.isOpened()):
        ret, imData = cap.read()
        if ret:
            imData = cv2.cvtColor(imData, cv2.COLOR_RGB2GRAY)
            if nFrame == 0:
                templatelist = [getTemplate(imData, roi) for roi in roilist]
                print('Got the templates')
            legCoords[nFrame] = trackAllTemplates(templatelist, roilist, imArgs, imData)
            if nFrame%600 == 0:
                sys.stdout.write("\r%s: %d"%(present_time(),nFrame))
                sys.stdout.flush()
            rois = getTrackedRois(roilist, legCoords[nFrame], imArgs['trackImSpread'])
            displayImageWithROI('Displaying Tracked Legs', cv2.cvtColor(imData, cv2.COLOR_GRAY2RGB), rois, imArgs)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('p'):
                print('\nPressed "p" at frame #%d'%nFrame)
                roilist = displayVid(vfname, roilist, imArgs, displayfps)
            nFrame += 1
        else:
            cap.release()
            cv2.destroyAllWindows()
    logFileWrite(logfname, present_time(), printContent = False)
    logFileWrite(logfname, '----------------------', printContent = False)
    values = legCoords
    return values
  





























import math
import matplotlib.pyplot as plt
import subprocess

def findOffset(data):
    '''
    returns the array of offset values for 'data'. This offset tells the
    most frequent position of the leg in the data.
    '''
    offset = np.zeros((len(data[0,:])))
    for leg in range (0,len(data[0,:])):
        counts = np.bincount(data[:,leg])
        offset[leg] = np.argmax(counts)#most frequent point in the data taken as offset
    return offset


def calcAngles(data,offset):
    '''
    returns the array containing eucledian distances and angles calculated
    from the given X,Y values in the 'data'. It takes care of the offset,
    i.e. the most frequent position of the leg in the 'data'
    '''
    angles = np.zeros((data.shape[0],data.shape[1]),dtype='float64')
    for leg in range(0,data.shape[1],2):
        for frame in range (0, data.shape[0]):
            x = data[frame,leg]-offset[leg]
            y = data[frame,leg+1]-offset[leg+1]
            if (x==0 or y==0 ):
                continue
            else:
                angles[frame,(leg)] = math.sqrt(math.pow(x,2)+math.pow(y,2))
                angles[frame,((leg)+1)] = math.degrees(math.atan2(y,x))
    return angles

def csvToData(data, step, anglesFileName):
    '''
    feed 'data' with x,y coordinates and get output as array with:
    Eucledian distance, angle of movement in degrees
    '''
    angles = np.zeros((len(data),len(data[0,:])),dtype='float64')
    segments = int(len(data)/step)
    for segment in range(0,segments):
        dataSegment = data[segment*step:(segment+1)*step,:]
        offset = findOffset(dataSegment)
        anglesSegment = calcAngles(dataSegment, offset)
        angles[segment*step:(segment+1)*step,:] = anglesSegment
    np.savetxt(anglesFileName, angles, fmt='%-7.2f', delimiter = ',')
    return angles

def eucDistSubPlotProps(ax, color, Leglabel):
    '''
    function used to set subplot properties in the function 'plotDistance'
    '''
    ax.set_yticklabels((Leglabel+'   .',20,40,60,80))
    for n, tl in enumerate(ax.yaxis.get_ticklabels()):
        if n==0:
            tl.set_color(color)
        else:
            tl.set_color('k')
    return ax
    
def plotDistance(data, titletxt, plotName):
    '''
    plots the eucledian distance using 'data' and saves the plot with
    'plotName' and title as 'titletxt'
    '''
    global ax
    for i in range(0,5):
        nPlot = 515 - i
        ax = plt.subplot(nPlot)
        ax.set_yticks((0,10,20,30,40))
        if i==0:
            color ='blue'
            ax = eucDistSubPlotProps(ax, color, 'L1')
            ax.set_xticks((20000,40000,60000,80000,100000,120000))
            ax.set_xticklabels((200,400,600,800,1000,1200))
            plt.xlabel('time (Seconds)')
            
        elif i==1:
            color ='green'
            ax = eucDistSubPlotProps(ax, color, 'R1')
            ax.set_xticklabels(())
            ax.title.set_visible(False)
            ax.set_ylabel('Distance (um)')
        elif i==2:
            color ='red'
            ax = eucDistSubPlotProps(ax, color, 'L2')
            ax.set_xticklabels(())
            ax.title.set_visible(False)
        elif i==3:
            color ='cyan'
            ax = eucDistSubPlotProps(ax, color, 'R2')
            ax.set_xticklabels(())
            ax.title.set_visible(False)
        elif i==4:
            color ='black'
            ax = eucDistSubPlotProps(ax, color, 'BG')
            ax.set_xticklabels(())
            ax.title.set_visible(False)
        plt.plot(data[:,2*i], color = color)
        plt.subplots_adjust(hspace = .001)
        plt.suptitle(titletxt)
        plt.xlim(0,120000)
        plt.ylim(0,60)
    plt.savefig(plotName,dpi=300)
    plt.close()


def createTar(folderName, inputDir, outputDir):
    '''
    takes Inputs as:
    folderName  : the folder to be compressed
    inputDir    : the directory containing the folder
    outputDir   : the directory for output of the tar
    '''
    folder = inputDir+folderName
    f = open(outputDir+folderName+'.tar', 'w')
    size = subprocess.check_output(['du','-sb', folder]).split('\t')[0]
    tar = subprocess.Popen(['tar', '-cf', '-', folder, '--remove-files'], stdout=subprocess.PIPE)
    pv = subprocess.Popen(['pv','-s',size], stdin=tar.stdout, stdout=f)
    out, err = pv.communicate()
    if err:
        print(err)
    f.close()



def getImArgs(dirname, nThreads, pupaDetails, procInputFileType):
    '''
    returns the dictionary of arguments for image processing
    '''
    imResizeFactor = 0.5        #resize display image to this factor
    templateResizeFactor = 4    #resize template image to this factor for display
    nLegs = 4                   # no. of legs to be tracked
    nBgs = 2                    # no. of background templates to be tracked
    trackImSpread = 30          #number of pixles around ROI where the ROI will be tracked
    csvStep = 1000              # stepsize for getting the reference position of the leg from the tracked csv file
    if pupaDetails in ['', None, False]:
        genotype = 'processTmp'
    else:
        genotype = pupaDetails.split(' -')[0]
    templateKeyDict = {1: 'leg_L1', 2: 'leg_R1', \
                       3: 'leg_L2', 4: 'leg_R2', \
                       5: 'Background_1', 6: 'Background_2',\
                       }        
    dirname = getFolder(dirname, 'Select Input Directory with %s'%procInputFileType)
    imArgs = {'nLegs': nLegs,
              'roiColors': [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)], 
              'roiBorderThickness': 2, 
              'nRois': nLegs+nBgs,
              'imResizeFactor': imResizeFactor,
              'templateResizeFactor': templateResizeFactor,
              'imfolder': 'imageData',
              'roifolder': 'roi',
              'csvfolder': 'csv',
              'trackImSpread' : trackImSpread,
              'csvStep': csvStep, 
              'roiVal' : [504, 510, 110, 116],
              'templateKeys': templateKeyDict,
              'nThreads': nThreads
               }
    imArgs['baseDir'], imArgs['imDir'], \
    imArgs['roiDir'], imArgs['csvDir'] = createDirsCheck(dirname, genotype, \
                                                        imArgs['imfolder'],\
                                                        imArgs['roifolder'], \
                                                        imArgs['csvfolder'], \
                                                        baseDir = False, \
                                                        imDir = False, \
                                                        roiDir = True, \
                                                        csvDir = True)
    imArgs['logfname'] = os.path.join(imArgs['baseDir'], ('ProcessLog_%s.txt'%procInputFileType))
    
    imArgs['fType'] = procInputFileType
    if imArgs['fType'] in ['Tar', 'TAR', 'tar']:
        imArgs['procType'] = processTarParallel
        imArgs['fExtension'] = '.tar'
    if imArgs['fType'] in ['Avi', 'AVI', 'avi']:
        imArgs['procType'] = processAvi
        imArgs['fExtension'] = '.avi'
    if imArgs['fType'] in ['Imfolder', 'imFolder', 'ImFolder', 'IMFOLDER', 'imfolder']:
        imArgs['procType'] = processImFolders
        imArgs['fExtension'] = '.jpeg'
        
    return imArgs























