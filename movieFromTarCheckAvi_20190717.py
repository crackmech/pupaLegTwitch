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
import threading as th
import imageLegTwitchBaseFunctions_tmp_20190717_flyPC as imLegTw



def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def makeDirs(dirPath, printArg, printEnd='\n'):
    try:
        os.makedirs(dirPath)
    except FileExistsError:
        print(printArg, end=printEnd)
        #print('Output directory already exists')

def getDirList(inDir, getAbsolutePath = True):
    if getAbsolutePath:
        return natural_sort([ os.path.join(inDir,name) for name in os.listdir(inDir) \
                                 if os.path.isdir(os.path.join(inDir, name))])
    else:
        return natural_sort([name for name in os.listdir(inDir) \
                                 if os.path.isdir(os.path.join(inDir, name))])

def getFileList(inDir, fileExt):
    return natural_sort([ os.path.join(inDir,name) for name in os.listdir(inDir) \
                            if os.path.isfile(os.path.join(inDir, name)) and fileExt in name])


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

def tarReadtoDict(tarName, nCurThrds):
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
    
def DictToMovFFMPEG(imDataDict, fps, nThreads, codec, outFname):
    '''
    Writes the image data in imDataDict to a movie using subprocess 'ffmpeg'
    '''
    writeTime = time.time()
    print('Writing AVI file %s by %s'%(outFname, th.currentThread().getName()))
    imNames = natural_sort(imDataDict.keys())
    ffmpegCmd = ffmpegCommand(fps, nThreads, codec=codec, outfname=outFname)
    pipe = sp.Popen(ffmpegCmd, stdin=sp.PIPE)
    for i,f in enumerate(imNames):
        im = imDataDict[f]
        pipe.stdin.write(im) # https://gist.github.com/waylan/2353749
    pipe.stdin.close()
    print('Wrote %s at %s in: %.02f Seconds by %s'%
            (outFname, present_time(), (time.time()-writeTime), th.currentThread().getName()))
    thread_available.set()

#baseDir ='/media/data_ssd/'

inDir ='/media/fly/ncbsStorage/twitchData'
outDir = '/media/data_ssd/rawMovies'

#inDir ='/media/data_ssd/tmp'
#outDir = '/media/data_ssd/rawMovies_'

nThreads = 8
nSubProcess = 4


fps = 100
codec = 'libx264'
imFolder = 'imageData'
tarExt = '.tar'
movExt = '.avi'

print('----- Started at %s'%present_time())

nCurThrds = th.active_count()
thread_available = th.Event()

i = 0
genotypeDirs = getDirList(inDir, getAbsolutePath = True)
for gtDir in genotypeDirs:
    for d in getDirList(gtDir, getAbsolutePath = True):
        if imFolder in getDirList(d, getAbsolutePath = False):
            baseDir = os.path.join(d, imFolder)
            tarList = getFileList(baseDir, tarExt)
            if len(tarList)>0:
                outMovDir = os.path.join(outDir, *(baseDir.split(os.sep)[-3:]))
                if os.path.isdir(outMovDir):
                    print('--------- Processing directory: %s ---------'%d)
    #                makeDirs(outMovDir, printArg = 'Output directory already exists')
                    outFlist = imLegTw.getFiles(outMovDir, '*.avi')
                    for tarFname in tarList:
                        fname = tarFname.split(os.sep)[-1].split('.')[0]
                        outFname = os.path.join(outMovDir, fname+movExt)
                        if outFname not in outFlist:
                            i += 1
                            print(i, tarFname)
#                            pipe = sp.run("tar -tf '%s' | wc -l"% tarFname, stdout=sp.PIPE, shell=True)
#                            print(tarFname, '\n', outFname, '\n', fname)
#                            makeDirs(outMovDir, printArg='', printEnd='')
#                            nCurThrds = th.active_count()
#                            if nCurThrds>=nSubProcess:
#                                print('\n====> Waiting for a thread to finish, Total threads running: %d , --- at %s'\
#                                        %(nCurThrds, present_time()))
#                                thread_available.wait()
#                            imStackDict = tarReadtoDict(tarFname, nCurThrds)
#                            t = th.Thread(target = DictToMovFFMPEG, args = (imStackDict, fps, nThreads, codec, outFname, ))
#                            t.start()
#                            thread_available.clear()
#t.join()


a = '/media/fly/ncbsStorage/twitchData/P0163_noTempShift/20170407_013754_P0163xTrpA1/imageData/20170408_123753.tar'


#genotypeDirs = getDirList(inDir, getAbsolutePath = True)
#for gtDir in genotypeDirs:
#    for d in getDirList(gtDir, getAbsolutePath = True):
#        if imFolder in getDirList(d, getAbsolutePath = False):
#            baseDir = os.path.join(d, imFolder)
#            tarList = getFileList(baseDir, tarExt)
#            if len(tarList)>0:
#                outMovDir = os.path.join(outDir, *(baseDir.split(os.sep)[-3:]))
#                print('--------- Processing directory: %s ---------'%d)
#                makeDirs(outMovDir, printArg = 'Output directory already exists')
#                for tarFname in tarList:
#                    outFname = os.path.join(outMovDir, tarFname.split(os.sep)[-1].split('.')[0]+movExt)
#                    makeDirs(outMovDir, printArg='', printEnd='')
#                    nCurThrds = th.active_count()
#                    if nCurThrds>=nSubProcess:
#                        print('\n====> Waiting for a thread to finish, Total threads running: %d , --- at %s'\
#                                %(nCurThrds, present_time()))
#                        thread_available.wait()
#                    imStackDict = tarReadtoDict(tarFname, nCurThrds)
#                    t = th.Thread(target = DictToMovFFMPEG, args = (imStackDict, fps, nThreads, codec, outFname, ))
#                    t.start()
#                    thread_available.clear()
#t.join()





