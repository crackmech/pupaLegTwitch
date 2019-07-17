# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:37:03 2019

@author: fly
"""

import tarfile
#import numpy as np
#import cv2
import time
import subprocess as sp
import threading as th


import imageLegTwitchBaseFunctions_tmp_20190713 as imLegTw
import sys
import numpy as np
import os
import glob

try: input = raw_input
except NameError: pass


def makeDirs(dirPath, printArg, printEnd='\n'):
    try:
        os.makedirs(dirPath)
    except FileExistsError:
        print(printArg)#, end=printEnd)
        #print('Output directory already exists')

def getDirList(inDir, getAbsolutePath = True):
    if getAbsolutePath:
        return imLegTw.natural_sort([ os.path.join(inDir,name) for name in os.listdir(inDir) \
                                 if os.path.isdir(os.path.join(inDir, name))])
    else:
        return imLegTw.natural_sort([name for name in os.listdir(inDir) \
                                 if os.path.isdir(os.path.join(inDir, name))])

def getFileList(inDir, fileExt):
    return imLegTw.natural_sort([ os.path.join(inDir,name) for name in os.listdir(inDir) \
                            if os.path.isfile(os.path.join(inDir, name)) and fileExt in name])


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
            tarName, imLegTw.present_time(), (time.time()-readTime), nCurThrds))
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
    imNames = imLegTw.natural_sort(imDataDict.keys())
    ffmpegCmd = ffmpegCommand(fps, nThreads, codec=codec, outfname=outFname)
    pipe = sp.Popen(ffmpegCmd, stdin=sp.PIPE)
    for i,f in enumerate(imNames):
        im = imDataDict[f]
        pipe.stdin.write(im) # https://gist.github.com/waylan/2353749
    pipe.stdin.close()
    print('Wrote %s at %s in: %.02f Seconds '%(outFname, imLegTw.present_time(), (time.time()-writeTime)))
    thread_available.set()

def DictToTrackLeg(imDataDict, roilist, nThreads, outCsvFname):
    '''
    Writes the image data in imDataDict to a movie using subprocess 'ffmpeg'
    '''
    writeTime = time.time()
    imNames = imLegTw.natural_sort(imDataDict.keys())
    
    
    
    
    ffmpegCmd = ffmpegCommand(fps, nThreads, codec=codec, outfname=outFname)
    pipe = sp.Popen(ffmpegCmd, stdin=sp.PIPE)
    for i,f in enumerate(imNames):
        im = imDataDict[f]
        pipe.stdin.write(im) # https://gist.github.com/waylan/2353749
    pipe.stdin.close()
    print('Wrote %s at %s in: %.02f Seconds '%(outFname, imLegTw.present_time(), (time.time()-writeTime)))
    thread_available.set()


saveImData = True
saveRoiImage = False

dirname = '/media/aman/data/'

imResizeFactor = 0.5 #resize image to this factor for display feed of the camera
templateResizeFactor = 4 #resize image to this factor for display of the template
nLegs = 4 # no. of legs to be tracked
nBgs = 2 # no. of background templates to be tracked
trackImSpread = 100  #number of pixles around ROI where the ROI will be tracked

genotype = 'VideoTmp'#pupaDetails.split(' -')[0]
imfolder = 'imageData'
roifolder = 'roi'
csvfolder = 'csv'

inDir ='/media/fly/ncbsStorage/twitchData'
outDir = '/media/data_ssd/rawMovies'
nThreads = 8
nSubProcess = 6


fps = 100
codec = 'libx264'
imFolder = 'imageData'
tarExt = '.tar'
movExt = '.avi'

print('----- Started at %s'%imLegTw.present_time())

nCurThrds = th.active_count()
thread_available = th.Event()



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
tarFlist = glob.glob(os.path.join(vidDir, tarExt))

for nVid, tarFname in enumerate(tarFlist):
    if nVid == 0:
        roiList = imLegTw.roiSelVid(tarFname, roiList, imArgs, getROI = True)
    dirTime = imLegTw.present_time()
    tarName = tarFname.split(os.sep)[-1]
    imLegTw.logFileWrite(logFileName, "Video file : %s"%tarFname, printContent = True)
    trackedValues = imLegTw.decodeNProcParllel(tarFname, roiList, displayfps=100, imArgs = imArgs, nThreads=4)
    np.savetxt(os.path.join(csvDir, tarName+'_'+dirTime+"_XY.csv"), trackedValues, fmt = '%-7.2f', delimiter = ',')
    print("\r\nProcesing for video file %d/%d: "%(nVid, len(tarFname)))





genotypeDirs = getDirList(inDir, getAbsolutePath = True)
for gtDir in genotypeDirs:
    for d in getDirList(gtDir, getAbsolutePath = True):
        if imFolder in getDirList(d, getAbsolutePath = False):
            baseDir = os.path.join(d, imFolder)
            tarList = getFileList(baseDir, tarExt)
            if len(tarList)>0:
                outMovDir = os.path.join(outDir, *(baseDir.split(os.sep)[-3:]))
                print('--------- Processing directory: %s ---------'%d)
                makeDirs(outMovDir, printArg = 'Output directory already exists')
                for tarFname in tarList:
                    outFname = os.path.join(outMovDir, tarFname.split(os.sep)[-1].split('.')[0]+movExt)
                    makeDirs(outMovDir, printArg='', printEnd='')
                    nCurThrds = th.active_count()
                    if nCurThrds>=nSubProcess:
                        print('\n====> Waiting for a thread to finish, Total threads running: %d , --- at %s'\
                                %(nCurThrds, imLegTw.present_time()))
                        thread_available.wait()
                    imStackDict = tarFolderReadtoDict(tarFname, nCurThrds)
                    t = th.Thread(target = DictToMovFFMPEG, args = (imStackDict, fps, nThreads, codec, outFname, ))
                    t.start()
                    thread_available.clear()
t.join()





