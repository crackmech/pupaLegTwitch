#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:51:06 2019

@author: aman
"""
import imageLegTwitchBaseFunctions_tmp_20190717_flyPC as imLegTw
import multiprocessing as mp

dirname = '/media/data_ssd/tmp/CS/tmp_120k'
dirname = '/media/data_ssd/rawMovies'
nThreads = 4
pupaDetails = ''

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

cmdType = {#"CAMERA": 'cam',
           "AVI FILE":'avi',
           "TAR File": 'tar',
           "Images":'imfolder'}
fType = imLegTw.natural_sort(cmdType.keys())

root = tk.Tk()
v = tk.StringVar()  # variable for Selecting filetype from the RadioButton
nThrd = tk.StringVar()    # variable for the number of threads spinbox
error = tk.StringVar()  # variable to print error message


class FlashableLabel(tk.Label):
    def flash(self,count):
        bg = self.cget('background')
        fg = self.cget('foreground')
        self.configure(background=fg,foreground=bg)
        count +=1
        if (count < 10):
             self.after(200,self.flash, count) 

def ShowChoice():
    try:
        nThreads = int(thSelectSpinBox.get())
        if nThreads == 0:
            errorMsg = 'Select proper number of threads for processing'
            print(errorMsg)
            error.set(errorMsg)
            errorLabel.flash(0)
            return
        procInputFileType = v.get()
        print('Processing from: %s'%v.get())
        imArgs = imLegTw.getImArgs(dirname, nThreads, pupaDetails, procInputFileType)
        fileExt = imArgs['fExtension']
        processFn = imArgs['procType']
        print('Processing from the %s with #Threads: %d'%(imArgs['fType'], nThreads))
        root.destroy()
    except SystemExit:
        errorMsg = 'Select proper folder/method for processing'
        print(errorMsg)
        error.set(errorMsg)
        errorLabel.flash(0)
        return
    pool = mp.Pool(processes=nThreads)
    processFn(imArgs, fileExt, pool, displayTrackedIms = False)
    pool.close()





for val, fileType in enumerate(fType):
    radBtn = tk.Radiobutton(root, 
                   indicatoron = 0,
                   text = fileType,
                   padx = 20, 
                   variable = v, 
                   command = ShowChoice,
                   width = 10,
                   value = cmdType[fileType])
    radBtn.grid(row = 0, column = val+1)

fSelectLabel = tk.Label(root, text="""Process leg twitching from:""")
threadLabel = tk.Label(root, text="""# Threads for parallel processing:""")
erroBoxLabel = tk.Label(root, text="""Error:""")
errorLabel = FlashableLabel(root, textvariable = error)
error.set('no error')
thSelectSpinBox = tk.Spinbox(root, from_ = 0, to = mp.cpu_count(), width = 10, textvariable = nThrd, increment = 2)



nThrd.set(str(nThreads))
fSelectLabel.grid(row = 0, column = 0)
threadLabel.grid(row = 1, column = 0)
thSelectSpinBox.grid(row = 1, column = 1)
erroBoxLabel.grid(row = 10, column = 0)
errorLabel.grid(row = 10, column = 1)
root.mainloop()
print('done')



root.mainloop()




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







