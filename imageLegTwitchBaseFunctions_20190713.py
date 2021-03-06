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

#import time
#import math
#import matplotlib.pyplot as plt



saveImData = True

try:
    saveImData = sys.argv[1].lower() == 'true'
except:
    pass


dirname = '/home/pointgrey/imaging/'
#dirname = '/media/pointgrey/4TB_2/'
secondaryDisk = '/media/pointgrey/shared/'

home = '/home/pointgrey/'

dirname = '/media/aman/data/'

try:
    startSegment = sys.argv[1]
except:
    pass

saveRoiImage = False

imData = []
#roiVal = np.zeros((6,4))
values = []
img = []
img2 = []

roiVal = [504, 510, 110, 116]
templateVal =  np.array([[10, 11, 13, 10, 10, 10], \
                         [ 9, 10, 12, 10, 11, 11], \
                         [10, 11, 12, 12, 10, 10], \
                         [11, 10, 11, 11, 11, 11], \
                         [11, 10, 12, 10, 11, 10], \
                         [10, 10, 10, 10, 12, 10]],dtype = np.uint8)

roiList = [roiVal]
templateList = [templateVal]

roiColors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
roiBorderThickness = 2

templateKeyDict = {1: 'leg L1', \
                   2: 'leg R1', \
                   3: 'leg L2', \
                   4: 'leg R2', \
                   5: 'Background-1',\
                   6: 'Background-2',\
                   }        


imfolder = 'imageData'
roifolder = 'roi'
csvfolder = 'csv'

imResizeFactor = 0.5 #resize image to this factor for display feed of the camera
nLegs = 4 # no. of legs to be tracked
nBgs = 2 # no. of background templates to be tracked

templateResizeFactor = 3 #resize image to this factor for display of the template

nRois = nLegs+nBgs

roiSelKeys = [ord(str(x)) for x in range(nRois)] # keys to be pressed on keyboard for selecting the roi



def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

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
    if imData.shape[-1]<3:
        return template
    else:
        return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

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




#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#

def roiSel(roilist, imArgs, getROI = True):
    '''
    Standalone function to select ROIs from an image captured by the camera
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
    #Capture image
    c.retrieve_buffer(im)
    imData = np.array(im)
    #Select ROI from the captured image
    while(1):
        rois = selRoi(imData, roilist, imArgs, \
                       getROI, showRoiImage = True, saveRoiImage = False)
        if rois != "error":
            break
    c.stop_capture()
    c.disconnect()
    return rois

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
                templatelist = [getTemplate(imData, roi) for roi in roiList]
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
   
     
try: input = raw_input
except NameError: pass


pupaDetails = input('Enter the pupa details : <genotype> - <APF> - <time> - <date>\n')

genotype = pupaDetails.split(' -')[0]

imDuration = 20      #in minutes
trackImSpread = 30  #number of pixles around ROI where the ROI will be tracked
totalLoops = 500    #total number of times the imaging loop runs
#delay = 0           #in seconds, delay between capturing each loop of duration = imDuration
#step  = 1000        #step for updating resting position of the leg, to calculate eucDis

try:
    baseDir, imDir, roiDir, csvDir = createDirs(dirname, genotype, imfolder, roifolder, csvfolder)
except:
    print("No directories available, please check!!!")
    sys.exit()

logFileName = os.path.join(baseDir, "camloop.txt")

logFileWrite(logFileName, pupaDetails, printContent = False)
try:
    startSegment = getStartSeg()
except:
    startSegment = 0

roiList = selPreviousRois(roiDir, roiVal, nRois)

SaveDirDuration = int(120/imDuration)

nFrames = int((imDuration*60*100)+1)
nFrames = 601
print(nFrames)

logFileWrite(logFileName, '----------------------', printContent = False)
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

roiList = roiSel(roiList, imArgs, getROI = True)
roiFileName = roiDir+"ROIs_"+present_time()+".txt"
np.savetxt(roiFileName, roiVal, fmt = '%d', delimiter = ',')
os.chdir(imDir)

c = fc2.Context()
c.connect(*c.get_camera_from_index(0))
im = fc2.Image()
c.start_capture()
c.retrieve_buffer(im)
imData = np.array(im)
templateList = [getTemplate(imData, roi) for roi in roiList]

for nLoop in range (startSegment,totalLoops):
    dirTime = present_time()
    try:
        saveDir = os.path.join(imDir, dirTime)
        os.mkdir(saveDir)
    except:
        saveImData = False
        saveDir = os.path.join(home, dirTime)
        os.mkdir(saveDir)
    logFileWrite(logFileName, "Directory : "+saveDir, printContent = True)
    trackedValues = CapNProc(c, im, roiList, templateList, nFrames, saveDir, saveImData, imArgs)
    np.savetxt(os.path.join(csvDir, dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
    print("\r\nProcesing for loop number: "+str(nLoop+1))


c.stop_capture()
c.disconnect()

logFileWrite(logFileName, '----------------------', printContent = False)
#
#

