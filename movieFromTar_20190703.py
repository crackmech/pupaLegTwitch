# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:37:03 2019

@author: fly
"""

import tarfile
import os
#import numpy as np
#import cv2
import time
import subprocess as sp
from datetime import datetime
import re
from threading import Thread
import threading as th



def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

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

def tarReadtoDict(tarName):
    '''
    read contents of the imageData tar folder into a dict
    '''
    print('Reading tar file from %s'%tarName)
    readTime = time.time()
    tar = tarfile.open(tarName,'r|') 
    tarStack = {}
    for f in tar:
        if f.isfile():
            c = tar.extractfile(f).read()
            fname = f.get_info()['name']
            tarStack[fname] = c
    tar.close()
    print('Read %s at %s in: %.02f Seconds '%(tarName, present_time(), (time.time()-readTime)))
    return tarStack
    
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
    #print('Total threads running: %d'%len(th.enumerate()))



baseDir ='/media/data_ssd/'

inDir ='/media/fly/ncbsStorage/twitchData'
outDir = '/media/data_ssd/rawMovies'

#inDir ='/media/data_ssd/tmp'
#outDir = '/media/data_ssd/rawMovies_'

fps = 100
codec = 'libx264'
nThreads = 8
nSubProcess = 6


tarExt = '.tar'
movExt = '.avi'

print('----- Started at %s'%present_time())

def getDirList(inDir):
    return natural_sort([ os.path.join(inDir,name) for name in os.listdir(inDir) \
                             if os.path.isdir(os.path.join(inDir, name))])

def getFileList(inDir, fileExt):
    return natural_sort([ os.path.join(inDir,name) for name in os.listdir(inDir) \
                            if os.path.isfile(os.path.join(inDir, name)) and fileExt in name])
                            
imFolder = 'imageData'

genotypeDirs = getDirList(inDir)
for gtDir in genotypeDirs:
    dirList = getDirList(gtDir)
    for d in dirList:
        folderList = os.listdir(d)
        if imFolder in folderList:
            baseDir = os.path.join(d, imFolder)
            tarList = getFileList(baseDir, tarExt)
            if len(tarList)>0:
                outMovDir = os.path.join(outDir, *(baseDir.split(os.sep)[-3:]))
                #print(baseDir)
                #print(outMovDir)
                print('--------- Processing directory: %s ---------'%d)
                try:
                    os.makedirs(outMovDir)
                except FileExistsError:
                    print('Output directory already exists')
                for tarFname in tarList:
                    outFname = os.path.join(outMovDir, tarFname.split(os.sep)[-1].split('.')[0]+movExt)
                    #print(tarFname,'\n',outFname, '\n')
                    try:
                        os.makedirs(outMovDir)
                    except FileExistsError:
                        pass
                    imStackDict = tarReadtoDict(tarFname)
                    t = th.Thread(target = DictToMovFFMPEG, args = (imStackDict, fps, nThreads, codec, outFname, ))
                    t.start()
                    if len(th.enumerate())>=nSubProcess:
                        print('====> Waiting for a thread to finish, Total threads running: %d , --- at %s '%(len(th.enumerate()), present_time()))
                        t.join()
                        print('====> Finished thread, Total threads running: %d , --- at %s '%(len(th.enumerate()), present_time()))
t.join()




#tarDir = '20161011_CS_data/imageData'
#baseDir = os.path.join(inDir, tarDir)
#outMovDir = os.path.join(outDir, *(baseDir.split(os.sep)[-3:]))
#try:
#    os.makedirs(outMovDir)
#except FileExistsError:
#    print('Output directory already exists')
#
#tarList = natural_sort([ os.path.join(baseDir,name) for name in os.listdir(baseDir) \
#                         if os.path.isfile(os.path.join(baseDir, name)) and tarExt in name])
#for tarFname in tarList:
#    outFname = os.path.join(outMovDir, tarFname.split(os.sep)[-1].split('.')[0]+movExt)
#    try:
#        os.makedirs(outMovDir)
#    except FileExistsError:
#        pass
#    imStackDict = tarReadtoDict(tarFname)
#    t = Thread(target = DictToMovFFMPEG, args = (imStackDict, fps, nThreads, codec, outFname, ))
#    t.start()
#



