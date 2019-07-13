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
import time
import tkFileDialog as tkd
import Tkinter as tk
import math
#import matplotlib.pyplot as plt



saveImDir = False

try:
    saveImDir = sys.argv[1].lower() == 'true'
except:
    pass


dirname = '/home/pointgrey/imaging/'
#dirname = '/media/pointgrey/4TB_2/'
secondaryDisk = '/media/pointgrey/shared/'

try:
    startSegment = sys.argv[1]
except:
    pass

drawing = False # true if mouse is pressed
saveRoiImage=False
ix,iy = -1,-1

imData=[]
roiVal=np.zeros((6,4))
values = []
img = []
img2 = []

roiVal1 = np.array([504, 510, 110, 116])
templateVal =  \
np.array([[10, 11, 13, 10, 10, 10],
       [ 9, 10, 12, 10, 11, 11],
       [10, 11, 12, 12, 10, 10],
       [11, 10, 11, 11, 11, 11],
       [11, 10, 12, 10, 11, 10],
       [10, 10, 10, 10, 12, 10]],dtype='uint8')
roi_l1 = roiVal1.copy
roi_r1 = roiVal1.copy
roi_l2 = roiVal1.copy
roi_r2 = roiVal1.copy
roi_bg = roiVal1.copy
roi_bg1 = roiVal1.copy

template_l1_gray = templateVal.copy()
template_r1_gray = templateVal.copy()
template_l2_gray = templateVal.copy()
template_r2_gray = templateVal.copy()
template_bg_gray = templateVal.copy()
template_bg1_gray = templateVal.copy()


def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

def logFileWrite(content):
    '''Create log files'''
    try:
        logFile = open(logFileName,'a')# Trying to create a new file or open one
        logFile.write('\n')
        logFile.write(content)
        logFile.close
    except:
        print('Something went wrong! Can\'t create log file')

def getStartSeg():
    '''
    gets the segment value for starting the imaging loop from this value.
    This can be used to start the imaging loop if imaging is interupted due
    to any reason
    '''
    try:
        startSegment=sys.argv[1]
    except:
        print 'Enter starting segment value : '
        startSegment = raw_input()
    return int(startSegment)

def createDirs(dirname, genotype):
    '''
    creates directories for saving images and csvs
    '''
    #create base directory for all data using current date as the foldername
    try:
        presentDate = present_time()
        os.mkdir(dirname+presentDate+'_'+genotype+'/')
        baseDir = dirname+presentDate+'_'+genotype+'/'
    except:
        print "Not able to create directories"
        pass
    try:
        #create directory for saving captured images (saved every 2 hours)  in the base directory
        imDir=baseDir+'imageData/'
        os.mkdir(imDir)
        #create directory for saving ROI images and ROI files in the base directory
        roiDir = baseDir+'roi/'
        os.mkdir(roiDir)
        csvDir = baseDir+'csv/'
        os.mkdir(csvDir)
        return baseDir, imDir, roiDir, csvDir
    except:
        pass

def selPreviousRois():
    '''
    returns the ROIs by reading the ROI values from a previous ROI file
    '''
    try:
        tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename = tkd.askopenfilename(initialdir = roiDir, filetypes=[("Text files","*.txt")]) # show an "Open" dialog box and return the path to the selected file
        print "Using ROIs from :"+filename
        rois = np.genfromtxt(filename,dtype='uint16',delimiter=',')
        roi_l1 = rois[0,:]
        roi_r1 = rois[1,:]
        roi_l2 = rois[2,:]
        roi_r2 = rois[3,:]
        roi_bg = rois[4,:]
        roi_bg1 = rois[5,:]
    except:
        print "Not using previously selected ROIs, select new ROIs"
        roi_l1 = roiVal1.copy
        roi_r1 = roiVal1.copy
        roi_l2 = roiVal1.copy
        roi_r2 = roiVal1.copy
        roi_bg = roiVal1.copy
        roi_bg1 = roiVal1.copy
    return roi_l1, roi_r1, roi_l2, roi_r2, roi_bg, roi_bg1

def resizeImage(imData):
    '''
    resizes the image to half of the original dimensions
    '''
    newx,newy = imData.shape[1]/2,imData.shape[0]/2 #new size (w,h)
    resizedImg = cv2.resize(imData,(newx,newy))
    return(resizedImg)

def ShowImage(imData):
    '''
    Loads and displays nth image from the current directory
    '''
    cv2.imshow("Image, Press Escape to kill Window",resizeImage(imData))

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

def ShowImageWithROI(imData, showRoiImage=True, saveRoiImage=False):
    '''
    Displays the imData with ROIs that are selected.
    If saveRoiImage is true, the image with ROIs marked on it is saved
    '''
    try:
        img = resizeImage(imData)
        cv2.rectangle(img, (roi_l1[0]/2, roi_l1[2]/2), (roi_l1[1]/2, roi_l1[3]/2), (255,0,0), 2)
        cv2.rectangle(img, (roi_r1[0]/2, roi_r1[2]/2), (roi_r1[1]/2, roi_r1[3]/2),(0,255,0), 2)
        cv2.rectangle(img, (roi_l2[0]/2, roi_l2[2]/2), (roi_l2[1]/2, roi_l2[3]/2),(0,0,255), 2)
        cv2.rectangle(img, (roi_r2[0]/2, roi_r2[2]/2), (roi_r2[1]/2, roi_r2[3]/2), (255,255,0), 2)
        cv2.rectangle(img, (roi_bg[0]/2, roi_bg[2]/2), (roi_bg[1]/2, roi_bg[3]/2), (255,0,255), 2)
        cv2.rectangle(img, (roi_bg1[0]/2, roi_bg1[2]/2), (roi_bg1[1]/2, roi_bg1[3]/2), (255,0,255), 2)
        
        if saveRoiImage==True:
            roiImageName = roiDir+present_time()+"_ROI.jpeg"
            cv2.imwrite(roiImageName,img)
            logFileWrite('Saved ROI Image as : '+ roiImageName)
            
        if showRoiImage==True:
            while(1):
                cv2.imshow("Selected ROIs, Press 'Esc' to exit, press'r' to reselect ROIs", img)
                k = cv2.waitKey(10) & 0xFF
                if k == 27:
                    break
                if k == ord("r"):
                    print "Selecting ROIs again"
                    cv2.destroyAllWindows()
                    selROI1(imData, getROI=True, updateROI=False, showRoiImage=True, saveRoiImage=False)
                    break
        cv2.destroyAllWindows()
    except:
        print 'No Folder / ROIs selected'

def getTemplate(imData,roi):
    '''
    Returns the template from the given image data using roi values
    '''
    template=(imData)[roi[2]:roi[3], roi[0]:roi[1]]
    return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

def selROI1(imData, getROI=True, updateROI=True, showRoiImage=True, saveRoiImage=True):
    '''
    Function for selecting the ROI on the image. Key press '1', '2', '3' and '4'
    saves the corresponding ROIs in as ROI for 'L1', 'R1', 'L2', 'L2' legs of 
    the pupa. Pressing '5' selects for the background.
    Boolean values dictate if you want to do what is described in the variable
    '''    
    global img, img2, roi_l1,roi_r1, roi_l2,roi_r2, roi_bg, roi_bg1, rect, dirlist,\
    template_l1_gray, template_r1_gray, template_l2_gray, template_r2_gray,\
    template_bg_gray, template_bg1_gray
    try:
        imDataColor=cv2.cvtColor(imData,cv2.COLOR_GRAY2BGR)
    except:
        imDataColor=imData
    if (getROI==True):
        try:
            rect=()
            img = resizeImage(imDataColor)
            img2 = img.copy()
            cv2.namedWindow("Select ROI, Press 'Esc' to exit")
            cv2.setMouseCallback("Select ROI, Press 'Esc' to exit",DrawROI)
            while(1):
                cv2.imshow("Select ROI, Press 'Esc' to exit",img)
                k = cv2.waitKey(10) & 0xFF
                if k == 27:
                    break
                if k == ord("1"):
                    roi_l1 = 2*(np.asarray(rect))
                    template_l1_gray=getTemplate(imDataColor,roi_l1)
                    cv2.imshow('L1_Template', template_l1_gray)
                    print 'L1 Template selected'
                if k == ord("2"):
                    roi_r1 = 2*(np.asarray(rect))
                    template_r1_gray=getTemplate(imDataColor,roi_r1)
                    cv2.imshow('R1_Template', template_r1_gray)
                    print 'R1 Template selected'
                if k == ord("3"):
                    roi_l2 = 2*(np.asarray(rect))
                    template_l2_gray=getTemplate(imDataColor,roi_l2)
                    cv2.imshow('L2_Template', template_l2_gray)
                    print 'L2 Template selected'
                if k == ord("4"):
                    roi_r2 = 2*(np.asarray(rect))
                    template_r2_gray=getTemplate(imDataColor,roi_r2)
                    cv2.imshow('R2_Template', template_r2_gray)
                    print 'R2 Template selected'
                if k == ord("5"):
                    roi_bg = 2*(np.asarray(rect))
                    template_bg_gray=getTemplate(imDataColor,roi_bg)
                    cv2.imshow('BG_Template', template_bg_gray)
                    print 'BG Template selected'
                if k == ord("6"):
                    roi_bg1 = 2*(np.asarray(rect))
                    template_bg1_gray=getTemplate(imDataColor,roi_bg1)
                    cv2.imshow('BG1_Template', template_bg1_gray)
                    print 'BG1 Template selected'
            cv2.destroyAllWindows()
            roiVal[0,:] = roi_l1
            roiVal[1,:] = roi_r1
            roiVal[2,:] = roi_l2
            roiVal[3,:] = roi_r2
            roiVal[4,:] = roi_bg
            roiVal[5,:] = roi_bg1
            roiFileName = roiDir+"ROIs_"+present_time()+".txt"
            np.savetxt(roiFileName, roiVal, fmt='%d', delimiter = ',')
        except ValueError:
            print "Select 'ALL' 4 ROIs"
            return "error"
        logFileWrite("Selected new ROIs")
    elif (updateROI==True):
        try:
            template_l1_gray=getTemplate(imDataColor,roi_l1)
            template_r1_gray=getTemplate(imDataColor,roi_r1)
            template_l2_gray=getTemplate(imDataColor,roi_l2)
            template_r2_gray=getTemplate(imDataColor,roi_r2)
            template_bg_gray=getTemplate(imDataColor,roi_bg)
            template_bg1_gray=getTemplate(imDataColor,roi_bg1)
        except:
            print "Error in updating templates"
            return "error"
    ShowImageWithROI(imDataColor, showRoiImage, saveRoiImage)

def trackTemplate(template,roi, imData, trackImSpread):
    '''
    Returns the XY values of the template after tracking in the given number of
    frames. Tracking is done using cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    function from opnCV library. 'template' is tracked in the 'imData' file.
    'template' is an array of values of selected ROI in grayscale. 
    'roi' is a list of X,Y locations and dimensions of the roi selected.
    'trackImSpread' is the number of pixels around roi where the template is tracked.
    '''
    w, h = template.shape[::-1]
    img = imData
    img1 = img[roi[2]-trackImSpread:roi[3]+trackImSpread, \
                    roi[0]-trackImSpread:roi[1]+trackImSpread]
    result = cv2.matchTemplate(img1,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc

def roiSel(getROI=True, updateROI=False):
    '''
    Standalone function to select ROIs from an image captured by the camera
    '''
    #Initiate camera
    c = fc2.Context()
    c.connect(*c.get_camera_from_index(0))
    print "roiSel : "+present_time()
    p = c.get_property(fc2.FRAME_RATE)
    print p
    c.set_property(**p)
    c.start_capture()
    im = fc2.Image()
    #Capture image
    c.retrieve_buffer(im)
    imData = np.array(im)
    #Select ROI from the captured image
    while(1):
        a=selROI1(imData, getROI, updateROI, showRoiImage=True, saveRoiImage=False)
        if a!="error":
            break
    c.stop_capture()
    c.disconnect()

def displayImageWithROI(windowName, imData):
    '''
    Continously display the images from camera with ROIs that are selected
    '''
    img = resizeImage(imData)
    cv2.rectangle(img, (roi_l1[0]/2, roi_l1[2]/2), (roi_l1[1]/2, roi_l1[3]/2), (255,0,0), 2)
    cv2.rectangle(img, (roi_r1[0]/2, roi_r1[2]/2), (roi_r1[1]/2, roi_r1[3]/2), (0,255,0), 2)
    cv2.rectangle(img, (roi_l2[0]/2, roi_l2[2]/2), (roi_l2[1]/2, roi_l2[3]/2), (0,0,255), 2)
    cv2.rectangle(img, (roi_r2[0]/2, roi_r2[2]/2), (roi_r2[1]/2, roi_r2[3]/2), (255,255,0), 2)
    cv2.rectangle(img, (roi_bg[0]/2, roi_bg[2]/2), (roi_bg[1]/2, roi_bg[3]/2), (255,0,255), 2)
    cv2.rectangle(img, (roi_bg1[0]/2, roi_bg1[2]/2), (roi_bg1[1]/2, roi_bg1[3]/2), (255,0,255), 2)
    cv2.imshow(windowName, img)

#---------------------------------------------------------------------------------#
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
        dataSegment = data[step*segment:(segment+1)*step,:]
        offset = findOffset(dataSegment)
        anglesSegment = calcAngles(dataSegment, offset)
        angles[step*segment:(segment+1)*step,:] = anglesSegment
    np.savetxt(anglesFileName, angles, fmt='%-7.2f', delimiter = ',')
    logFileWrite('Angles saved as: '+anglesFileName)
    return angles

def disAngTxtnPlot(data, step, fileName, plotTitle ):
    '''
    processes the XY 'data'  for calculating eucledian distance and angles;
     using function 'csvToData'
    '''
    csvToData(data, step, fileName+'_eucdisAngles.txt')

#---------------------------------------------------------------------------------#
def displayCam(c, im):
    '''
    Starts display from the connected camera. ROIs can be updated by pressing 'u'
    '''
    while(1):
        c.retrieve_buffer(im)
        imData = np.array(im)
        imDataColor=cv2.cvtColor(imData,cv2.COLOR_GRAY2BGR)
        windowName = "Camera Display, Press 'u' to update ROIs or 'Esc' to close"
        displayImageWithROI(windowName, imDataColor)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord("u"):
            selROI1(imData, getROI=True, updateROI=False, showRoiImage=True, saveRoiImage=True)
    cv2.destroyWindow(windowName)


def CapNProc(c, im, nFrames, saveDir, baseDir, csvDir, saveIm, dirTime):
    '''
    Function for capturing images from the camera and then tracking already defined
    templates. The template images are updated (using ROIs defined earlier)
    everytime the function is called. Pressing 'Ctrl+c' pauses the tracking loop
    and starts displaying live images from the camera. This can be used to select
    new templates while the function is running.
    '''
    print present_time()+" Press 'Ctrl+C' to pause analysis and start live display"
    l1=np.zeros((nFrames,2), dtype='uint16')
    r1=np.zeros((nFrames,2), dtype='uint16')
    l2=np.zeros((nFrames,2), dtype='uint16')
    r2=np.zeros((nFrames,2), dtype='uint16')
    bg=np.zeros((nFrames,2), dtype='uint16')
    bg1=np.zeros((nFrames,2), dtype='uint16')
    fileName = csvDir+dirTime+"_XY"
    logFileWrite(present_time())
    for nFrame in range (0,nFrames):
        try:
            if nFrame%100==0:
                sys.stdout.write("\r%s: %d"%(present_time(),nFrame))
                sys.stdout.flush()
            c.retrieve_buffer(im)
            imData = np.array(im)
            if nFrame==10:
                startNT(selROI1, (imData, False, True, False, True))
                if np.median(imData)>250: # to stop imaging if the image goes white predominantely
                    print "\n-------No pupa to Image anymore!!-------"
                    logFileWrite("------- No pupa to image anymore!! Imaging exited -------")
                    sys.exit(0)
#                selROI1(imData, getROI=False, updateROI=True, showRoiImage=False, saveRoiImage=True)
            l1[nFrame,:] = trackTemplate(template_l1_gray, roi_l1, imData, trackImSpread)
            r1[nFrame,:] = trackTemplate(template_r1_gray, roi_r1, imData, trackImSpread)
            l2[nFrame,:] = trackTemplate(template_l2_gray, roi_l2, imData, trackImSpread)
            r2[nFrame,:] = trackTemplate(template_r2_gray, roi_r2, imData, trackImSpread)
            bg[nFrame,:] = trackTemplate(template_bg_gray, roi_bg, imData, trackImSpread)
            bg1[nFrame,:] = trackTemplate(template_bg1_gray, roi_bg1, imData, trackImSpread)
            if saveIm==True:
                try:
                    startNT(cv2.imwrite,(saveDir+str(nFrame)+'.jpeg',imData,))
                except:
                    print "error saving "+str(nFrame)
            elif saveIm==False:
                if nFrame%1000==0:
                    startNT(cv2.imwrite,(saveDir+str(nFrame)+'.jpeg',imData,))
        except KeyboardInterrupt:
            print "\nCamera display started on "+present_time()
            logFileWrite("Camera display started on "+present_time())
            displayCam(c, im)
            print "Camera display exited on  "+present_time()
            logFileWrite("Camera display exited on  "+present_time())
    logFileWrite(present_time())
    logFileWrite('----------------------')
    values = np.c_[l1,r1,l2,r2,bg,bg1]
    try:
        np.savetxt(fileName+'.csv', values, fmt='%-7.2f', delimiter = ',')
    except:
        fileName = home+dirTime+"_XY"
        np.savetxt(fileName+'.csv', values, fmt='%-7.2f', delimiter = ',')
        

home = '/home/pointgrey/'
print 'Enter the pupa details : <genotype> - <APF> - <time> - <date>'
pupaDetails = raw_input()

genotype = pupaDetails.split(' -')[0]

imDuration = 20      #in minutes
delay = 0           #in seconds, delay between capturing each loop of duration = imDuration
trackImSpread = 30  #number of pixles around ROI where the ROI will be tracked
step  = 1000        #step for updating resting position of the leg, to calculate eucDis
totalLoops = 500    #total number of times the imaging loop runs

try:
    baseDir, imDir, roiDir, csvDir = createDirs(dirname, genotype)
except:
    print "No directories available, please check!!!"
    sys.exit()
logFileName = baseDir+"camloop.txt"



logFileWrite(pupaDetails)
try:
    startSegment = getStartSeg()
except:
    startSegment = 0
roi_l1, roi_r1, roi_l2, roi_r2, roi_bg, roi_bg1 = selPreviousRois()
SaveDirDuration = int(120/imDuration)
#variable for creating directory in loop
nFrames = int((imDuration*60*100)+1)
print nFrames
#nFrames = 701

ax = []
logFileWrite('----------------------')

roiSel(getROI=True, updateROI=False)
roiFileName = roiDir+"ROIs_"+present_time()+".txt"
np.savetxt(roiFileName, roiVal, fmt='%d', delimiter = ',')
os.chdir(imDir)


c = fc2.Context()
c.connect(*c.get_camera_from_index(0))
im = fc2.Image()
c.start_capture()

for nLoop in range (startSegment,totalLoops):
#    if nLoop==100 and saveImDir ==True:
#        dirname = secondaryDisk
#        baseDir, imDir, roiDir, csvDir = createDirs(dirname)
#        logFileName = baseDir+"camloop.txt"
    dirTime = present_time()
    try:
        saveDir = imDir+dirTime+'/'
        os.mkdir(saveDir)
        print "Direcory : "+saveDir
        logFileWrite("Direcory : "+saveDir)
        CapNProc(c, im, nFrames, saveDir, baseDir, csvDir, saveImDir, dirTime)
    except:
        saveImDir = False
        saveDir = home+dirTime+'/'
        os.mkdir(saveDir)
        print "Direcory : "+saveDir
        logFileWrite("Direcory : "+saveDir)
        CapNProc(c, im, nFrames, saveDir, baseDir, csvDir, saveImDir, dirTime)            
    print "\r\nProcesing for loop number: "+str(nLoop+1)
    time.sleep(delay)


c.stop_capture()
c.disconnect()

logFileWrite('----------------------')
#
#

