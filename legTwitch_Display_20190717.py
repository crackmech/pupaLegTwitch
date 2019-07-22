#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:51:06 2019

@author: aman
"""
import imageLegTwitchBaseFunctions_tmp_20190717_flyPC as imLegTw
import multiprocessing as mp

dirname = '/media/data_ssd/tmp/CS/tmp_120k'
nThreads = 32
pupaDetails = ''

#procInputFileType = 'tar'
##procInputFileType = 'avi'
##procInputFileType = 'imfolder'
#
#imArgs = imLegTw.getImArgs(dirname, nThreads, pupaDetails, procInputFileType)
#fileExt = imArgs['fExtension']
#processFn = imArgs['procType']
#print('Processing from the %s'%imArgs['fType'])
#
#pool = mp.Pool(processes=nThreads)
#processFn(imArgs, fileExt, pool, displayTrackedIms = False)
#pool.close()
#

import sys
if sys.version_info[0] < 3:
#    import tkFileDialog as tkd
    import Tkinter as tk
else:
#    import tkinter.filedialog as tkd
    import tkinter as tk

cmdType = {#"CAMERA": 'cam',
           "TAR File": 'tar',
           "AVI FILE":'avi',
           "Images":'imfolder'}
fType = imLegTw.natural_sort(cmdType.keys())

root = tk.Tk()
v = tk.StringVar()  # initializing the choice, i.e. Python



def ShowChoice():
    procInputFileType = v.get()
    print('Processing from: %s'%v.get())
    try:
        imArgs = imLegTw.getImArgs(dirname, nThreads, pupaDetails, procInputFileType)
        fileExt = imArgs['fExtension']
        processFn = imArgs['procType']
        print('Processing from the %s'%imArgs['fType'])
        
        pool = mp.Pool(processes=nThreads)
        processFn(imArgs, fileExt, pool, displayTrackedIms = False)
        pool.close()
    except SystemExit:
        pass

tk.Label(root, 
         text="""Process leg twitching from:""",
         justify = tk.LEFT,
         padx = 20).pack()

for val, fileType in enumerate(fType):
    tk.Radiobutton(root, 
                   indicatoron = 0,
                   text = fileType,
                   padx = 20, 
                   variable = v, 
                   command = ShowChoice,
                   value = cmdType[fileType]).pack(anchor = tk.N)


root.mainloop()












#import os
#import numpy as np
#import glob
#import cv2
#import itertools
##import matplotlib.pyplot as plt

#vfname = '/media/data_ssd/tmp/CS/tmp_120k/20161012_173506_120K_x264.avi'
#imArgs['imResizeFactor'] = 0.1
#roilist = imLegTw.selPreviousRois(imArgs['roiDir'], imArgs['roiVal'], imArgs['nRois'])
#pool = mp.Pool(processes=nThreads)
#legCoords = imLegTw.decodeNProcParllel(vfname, roilist, 100, imArgs, pool, nThreads)
#pool.close()
#fileName = vfname.split(os.sep)[-1]+'_Parallel'
#dirTime = imLegTw.present_time()
#np.savetxt(os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+"_XY.csv"), legCoords, fmt = '%-7.2f', delimiter = ',')
#eucDisData = imLegTw.csvToData(legCoords, imArgs['csvStep'], os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucdisAngles.txt'))
#imLegTw.plotDistance(eucDisData, fileName, os.path.join(imArgs['csvDir'], fileName+'_'+dirTime+'_XY_eucDis.png'))







