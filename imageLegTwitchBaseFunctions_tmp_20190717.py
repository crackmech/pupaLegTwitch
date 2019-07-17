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
import flycapture2 as fc2
import numpy as np
import cv2
from datetime import datetime
import sys
from thread import start_new_thread as startNT
import os
import tkFileDialog as tkd
import Tkinter as tk
import multiprocessing as mp
import itertools
import re
import subprocess as sp

import time
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
    roilist = ShowImageWithROI(imDataColor, roilist, imArgs, showRoiImage, saveRoiImage)
    return roilist

def ShowImageWithROI(imData, roilist, imArgs, showRoiImage = True, saveRoiImage = False):
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
            roiImageName = os.path.join(imArgs['roiDir'], present_time()+"_ROI.jpeg")
            cv2.imwrite(roiImageName,img)
            logFileWrite(imArgs['logfname'], 'Saved ROI Image as : '+ roiImageName, printContent = False)
            
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

def displayImageWithROI(windowName, imData, roilist, imArgs):
    '''
    Continously display the images from camera with ROIs that are selected
    '''
    for i, roi in enumerate(roilist):
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
    legCoords = np.zeros((nFrames,2*imArgs['nRois']), dtype = 'uint16')
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
            for i, template in enumerate(templatelist):
                legCoords[nFrame, i:i+2] = trackTemplate(template, roilist[i], imData, imArgs['trackImSpread'])
            #print('\ni: %d\n nFrame: %d\ntemplateList length: %d\n roiList length: %d'%(i,nFrame, len(templatelist), len(roilist)))
            if saveIm == True:
                try:
                    startNT(cv2.imwrite,(os.path.join(saveDir, str(nFrame)+'.jpeg'),imData,))
                except:
                    print("error saving %s"+str(nFrame))
            elif saveIm == False:
                if nFrame%1000 == 0:
                    startNT(cv2.imwrite,(os.path.join(saveDir, str(nFrame)+'.jpeg'),imData,))
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

def decodeNProc(vfname, roilist, displayfps, imArgs):
    '''
    returns the tracking data for the selected ROIs from the video file
    '''
    logfname = imArgs['logfname']
    logFileWrite(logfname, present_time(), printContent = False)
    print("%s Press 'p' to pause analysis and start live display"%present_time())
    cap = cv2.VideoCapture(vfname)
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    legCoords = np.zeros((nFrames,2*imArgs['nRois']), dtype = 'uint16')
    nFrame = 0
    while(cap.isOpened()):
#        ret = cap.grab()
        ret, imData = cap.read()
        if ret:
            imData = cv2.cvtColor(imData, cv2.COLOR_RGB2GRAY)
            if nFrame == 0:
                templatelist = [getTemplate(imData, roi) for roi in roilist]
                print('Got the templates')
            for i, template in enumerate(templatelist):
                legCoords[nFrame, i:i+2] = trackTemplate(template, roilist[i], imData, imArgs['trackImSpread'])
            nFrame += 1
            if nFrame%100 == 0:
                sys.stdout.write("\r%s: %d"%(present_time(),nFrame))
                sys.stdout.flush()
            k = displayIm('Template Tracking', resizeImage(imData, 0.1), 1)
            if k == ord('p'):
                print('\nPressed "p" at frame #%d'%nFrame)
                roilist = displayVid(vfname, roilist, imArgs, displayfps)
        else:
            cap.release()
            cv2.destroyAllWindows()
    logFileWrite(logfname, present_time(), printContent = False)
    logFileWrite(logfname, '----------------------', printContent = False)
    values = legCoords#np.c_[l1,r1,l2,r2,bg,bg1]
    return values
  
def trackAllTemplates(templatelist, roilist, imArgs, imData):
    '''
    '''
    trackCoords = np.zeros(2*len(templatelist), dtype=np.uint16)
    for i, template in enumerate(templatelist):
        trackCoords[i:i+2] = trackTemplate(template, roilist[i], imData, imArgs['trackImSpread'])
    return trackCoords
    
def procVideo((vfname, roilist, templatelist, frameStep, imArgs, group_number)):
    cap = cv2.VideoCapture(vfname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameStep * group_number)
    legCoords = np.zeros((frameStep,2*imArgs['nRois']), dtype = np.uint16)
    proc_frames = 0
    print('Started processing on processor: %d' %group_number)
    while proc_frames < frameStep:
        ret, imData = cap.read()
        if ret:
            imData = cv2.cvtColor(imData, cv2.COLOR_RGB2GRAY)
        legCoords[proc_frames] = trackAllTemplates(templatelist, roilist, imArgs, imData)
        proc_frames += 1
    cap.release()
    return legCoords

def decodeNProcParllel(vfname, roilist, displayfps, imArgs, nThreads):
    '''
    returns the tracking data for the selected ROIs from the video file
    '''
    logfname = imArgs['logfname']
    logFileWrite(logfname, present_time(), printContent = False)
    startTime = time.time()
    print("Started processing at: %s"%present_time())
    cap = cv2.VideoCapture(vfname)
    frameStep =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // nThreads)
    ret, imData = cap.read()
    if ret:
        imData = cv2.cvtColor(imData, cv2.COLOR_RGB2GRAY)
        templatelist = [getTemplate(imData, roi) for roi in roilist]
        mpArgs = itertools.izip(itertools.repeat(vfname), \
                                itertools.repeat(roilist), \
                                itertools.repeat(templatelist), \
                                itertools.repeat(frameStep), \
                                itertools.repeat(imArgs), \
                                range(nThreads)
                                )
        legCoords = mp.Pool(nThreads).map(procVideo, mpArgs)
        legCoordsStack = np.zeros((len(legCoords)*legCoords[0].shape[0], legCoords[0].shape[1]), dtype=np.uint16)
        for i, coords in enumerate(legCoords):
            legCoordsStack[i*frameStep:(i+1)*frameStep] = coords
        print(len(legCoords), legCoords[0].shape, legCoordsStack.shape, legCoordsStack.max(), legCoordsStack.min())
        print('Done processing in %0.2f Seconds, at: %s' %(time.time()-startTime,present_time()))
        cap.release()
        return legCoordsStack

import tarfile

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
    
def ffmpegCommand(fps, nThreads, codec, outfname):
    '''
    creates a ffmpeg command for subprocess
    '''
    command = [ 'ffmpeg',
            '-r', str(fps), # FPS of the output video file
            '-i', 'pipe:0', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-threads', str(nThreads), # define number of threads for parallel processing
            '-loglevel', 'error', # silence the output of ffmpeg to error only
            '-vcodec', codec, # specify the codec to be used for vidoe encoding
            '-y', # overwrite the existing file without asking
            outfname # name of the output file
            ] 
    return command

def DictToMovFFMPEG(imDataDict, fps, nThreads, codec, outFname):
    '''
    Writes the image data in imDataDict to a movie using subprocess 'ffmpeg'
    '''
    writeTime = time.time()
    imNames = natural_sort(imDataDict.keys())
    ffmpegCmd = ffmpegCommand(fps, nThreads, codec=codec, outfname=outFname)
    pipe = sp.Popen(ffmpegCmd, stdin=sp.PIPE)
    for i,f in enumerate(imNames):
        im = imDataDict[f]
        pipe.stdin.write(im) # https://gist.github.com/waylan/2353749
    pipe.stdin.close()
    print('Wrote %s at %s in: %.02f Seconds '%(outFname, present_time(), (time.time()-writeTime)))
    #thread_available.set()


def procImList((imNamesList, roilist, templatelist, frameStep, imArgs, group_number)):
    '''
    returns the coordinates of template matched ROIs from roilist using a specific thread
    '''
    imNamesListToProc = imNamesList[frameStep * group_number: (frameStep + 1) * group_number]
    legCoords = np.zeros((frameStep,2*imArgs['nRois']), dtype = np.uint16)
    #print('Started processing on thread: %d' %group_number)
    for nFrame, im in enumerate(imNamesListToProc):
        imData = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        legCoords[nFrame] = trackAllTemplates(templatelist, roilist, imArgs, imData)
    return legCoords

def parallelProcImList(imNamesList, roilist, imArgs, nThreads, pool):
    '''
    processes the imList using multiple threads
    '''
    frameStep =  int(len(imNamesList) // nThreads)
    imData = cv2.imread(imNamesList[0], cv2.IMREAD_GRAYSCALE)
    templatelist = [getTemplate(imData, roi) for roi in roilist]
    mpArgs = itertools.izip(itertools.repeat(imNamesList), \
                            itertools.repeat(roilist), \
                            itertools.repeat(templatelist), \
                            itertools.repeat(frameStep), \
                            itertools.repeat(imArgs), \
                            range(nThreads)
                            )
    return pool.map(procImList, mpArgs)
    
def procTarImList((imNamesList, roilist, templatelist, frameStep, imArgs, group_number)):
    '''
    returns the coordinates of template matched ROIs from roilist using a specific thread
    '''
    imNamesListToProc = imNamesList[frameStep * group_number: (frameStep + 1) * group_number]
    legCoords = np.zeros((frameStep,2*imArgs['nRois']), dtype = np.uint16)
    #print('Started processing on thread: %d' %group_number)
    for nFrame, im in enumerate(imNamesListToProc):
        imData = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        legCoords[nFrame] = trackAllTemplates(templatelist, roilist, imArgs, imData)
    return legCoords

def parallelProcTarIm(imNamesList, roilist, imArgs, nThreads, pool):
    '''
    processes the imList using multiple threads
    '''
    frameStep =  int(len(imNamesList) // nThreads)
    imData = cv2.imread(imNamesList[0], cv2.IMREAD_GRAYSCALE)
    templatelist = [getTemplate(imData, roi) for roi in roilist]
    mpArgs = itertools.izip(itertools.repeat(imNamesList), \
                            itertools.repeat(roilist), \
                            itertools.repeat(templatelist), \
                            itertools.repeat(frameStep), \
                            itertools.repeat(imArgs), \
                            range(nThreads)
                            )
    return pool.map(procTarImList, mpArgs)
    

def tarProcParllel(tarFname, roilist, imArgs, nThreads, pool):
    '''
    returns the tracking data for the selected ROIs from the tar file
    '''
    logFileWrite(imArgs['logfname'], present_time(), printContent = False)
    startTime = time.time()
    print("Started processing at: %s"%present_time())
    imNamesDict = tarFolderReadtoDict(tarName = tarFname, nCurThrds = 0)
    imNamesList = natural_sort(imNamesDict.keys())
    legCoords = parallelProcImList(imNamesList, roilist, imArgs, nThreads, pool)
    frameStep = legCoords[0].shape[0]
    legCoordsStack = np.zeros((len(legCoords)*legCoords[0].shape[0], legCoords[0].shape[1]), dtype=np.uint16)
    for i, coords in enumerate(legCoords):
        legCoordsStack[i*frameStep:(i+1)*frameStep] = coords
    print(len(legCoords), legCoords[0].shape, legCoordsStack.shape, legCoordsStack.max(), legCoordsStack.min())
    print('Done processing in %0.2f Seconds, at: %s' %(time.time()-startTime,present_time()))
    return legCoordsStack







# fis for tar imDict, it reads all files, not only names. use cv2.imdecode






































def decodeNProcParllel1(vfname, roilist, displayfps, imArgs):
    '''
    returns the tracking data for the selected ROIs from the video file
    '''
    logfname = imArgs['logfname']
    logFileWrite(logfname, present_time(), printContent = False)
    print("%s Press 'p' to pause analysis and start live display"%present_time())
    cap = cv2.VideoCapture(vfname)
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    imStack = np.zeros((nFrames, frameheight, frameWidth), dtype = np.uint8)
    nFrame = 0
    while(cap.isOpened()):
#        ret = cap.grab()
        ret, imData = cap.read()
        if ret:
            imData = cv2.cvtColor(imData, cv2.COLOR_RGB2GRAY)
            imStack[nFrame] = imData
            nFrame += 1
            if nFrame%100 == 0:
                sys.stdout.write("\r%s: %d"%(present_time(),nFrame))
                sys.stdout.flush()
        else:
            cap.release()
    legCoords = np.zeros((nFrames,2*imArgs['nRois']), dtype = 'uint16')
    print('\nread Images, now tracking at: %s'%present_time())
    for i, imData in enumerate(imStack):
        if i == 0:
            templatelist = [getTemplate(imData, roi) for roi in roilist]
            print('Got the templates')
        legCoords[i] = trackAllTemplates(templatelist, roilist, imArgs, imData)
    print('Tracked all images at: %s'%present_time())
#        for i, template in enumerate(templatelist):
#            legCoords[nFrame, i:i+2] = trackTemplate(template, roilist[i], imData, imArgs['trackImSpread'])
    return legCoords
#            k = displayIm('Template Tracking', resizeImage(imData, 0.1), 1)
#            if k == ord('p'):
#                print('\nPressed "p" at frame #%d'%nFrame)
#                roilist = displayVid(vfname, roilist, imArgs, displayfps)
#        else:
#            cap.release()
#            cv2.destroyAllWindows()
#    logFileWrite(logfname, present_time(), printContent = False)
#    logFileWrite(logfname, '----------------------', printContent = False)
#    values = legCoords#np.c_[l1,r1,l2,r2,bg,bg1]
#    return values
   
    

##    itertools.izip(flist, itertools.repeat(params), np.arange(len(flist)))
#    nFrame = 0
#    while(cap.isOpened()):
##        ret = cap.grab()
#        ret, imData = cap.read()
#        if ret:
#            imData = cv2.cvtColor(imData, cv2.COLOR_RGB2GRAY)
#            imStack[nFrame] = imData
#            nFrame += 1
#            if nFrame%100 == 0:
#                sys.stdout.write("\r%s: %d"%(present_time(),nFrame))
#                sys.stdout.flush()
#        else:
#            cap.release()
#    legCoords = np.zeros((nFrames,2*imArgs['nRois']), dtype = 'uint16')
#    print('\nread Images, now tracking at: %s'%present_time())
#    for i, imData in enumerate(imStack):
#        if i == 0:
#            templatelist = [getTemplate(imData, roi) for roi in roilist]
#            print('Got the templates')
#        legCoords[i] = trackAllTemplates(templatelist, roilist, imArgs, imData)
#    print('Tracked all images at: %s'%present_time())
##        for i, template in enumerate(templatelist):
##            legCoords[nFrame, i:i+2] = trackTemplate(template, roilist[i], imData, imArgs['trackImSpread'])
#    return legCoords
#


#            k = displayIm('Template Tracking', resizeImage(imData, 0.1), 1)
#            if k == ord('p'):
#                print('\nPressed "p" at frame #%d'%nFrame)
#                roilist = displayVid(vfname, roilist, imArgs, displayfps)
#        else:
#            cap.release()
#            cv2.destroyAllWindows()
#    logFileWrite(logfname, present_time(), printContent = False)
#    logFileWrite(logfname, '----------------------', printContent = False)
#    values = legCoords#np.c_[l1,r1,l2,r2,bg,bg1]
#    return values


     






