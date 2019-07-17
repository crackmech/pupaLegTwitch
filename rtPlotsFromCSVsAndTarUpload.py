#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:47:21 2017

@author: pointgrey
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:24:51 2016

@author: pointgrey
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 10:32:54 2016

@author: pointgrey
"""

'''
This script continously monitors the selected folder for the new csv file to
generate eucledian distance txt file and plot the eucledian distance graphs.

'''
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import re
import os
import Tkinter as tk
import tkFileDialog as tkd
import sys
import math
import time
import subprocess


ax = []

def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def ListFiles(pathName, extension):
    os.chdir (pathName)
    files = []
    for f in glob.glob(extension):
        files.append(f)
    return natural_sort(files)

def getFolder(initialDir, label):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select input directory')
    root.destroy()
    return initialDir+'/'

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
#    logFileWrite('Angles saved as: '+anglesFileName)
    return angles



def eucDistSubPlotProps(color, Leglabel):
    '''
    function used to set subplot properties in the function 'plotDistance'
    '''
    global ax
    ax.set_yticklabels((Leglabel+'   .',20,40,60,80))
    for n, tl in enumerate(ax.yaxis.get_ticklabels()):
        if n==0:
            tl.set_color(color)
        else:
            tl.set_color('k')
    
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
            eucDistSubPlotProps(color, 'L1')
            ax.set_xticks((20000,40000,60000,80000,100000,120000))
            ax.set_xticklabels((200,400,600,800,1000,1200))
            plt.xlabel('time (Seconds)')
            
        elif i==1:
            color ='green'
            eucDistSubPlotProps(color, 'R1')
            ax.set_xticklabels(())
            ax.title.set_visible(False)
            ax.set_ylabel('Distance (um)')
        elif i==2:
            color ='red'
            eucDistSubPlotProps(color, 'L2')
            ax.set_xticklabels(())
            ax.title.set_visible(False)
        elif i==3:
            color ='cyan'
            eucDistSubPlotProps(color, 'R2')
            ax.set_xticklabels(())
            ax.title.set_visible(False)
        elif i==4:
            color ='black'
            eucDistSubPlotProps(color, 'BG')
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




initialDir = '/home/pointgrey/imaging/'
outDir = '/media/pointgrey/shared/imageData/tarImageData/'

inDirname = getFolder(initialDir, 'Please select input directory')

outDir =  getFolder(outDir, 'Please select Tar output directory')

currDir =  inDirname.split('/')[-2]

os.makedirs(outDir+currDir+'/imageData/')
outDir = outDir+currDir+'/imageData/'
csvDir = inDirname+'csv/'
imDataDir = inDirname+'imageData/'

print("Input directory: %s"%inDirname)
print("Output directory: %s"%outDir)

csvList = ListFiles(csvDir, '*.csv')
txtList = ListFiles(csvDir, '*.txt')
step = 1000
if csvList!=[]:
    print(csvList)
if txtList!=[]:
    print(txtList)
print('Number of CSV files to process: %d'%len(csvList))
print('Number of txt files present: %d'%len(txtList))
for csvFile in csvList:
    if csvFile.rstrip('.csv')+'_eucdisAngles.txt' in txtList:
        print(csvDir+csvFile + ' already processed')
    elif csvFile.rstrip('.csv')+'_eucdisAngles.txt' not in txtList:
        print(csvDir+csvFile+ ' at ' +present_time())
        csvData = np.genfromtxt(csvDir+csvFile, delimiter=',', dtype = 'int16')
        data = csvToData(csvData, step, csvFile.rstrip('.csv')+'_eucdisAngles.txt')
        plotDistance(data, csvFile.rstrip('.csv'), csvFile.rstrip('.csv')+'_eucDis.png')
    
while (1):
    newCsvList = ListFiles(csvDir, '*.csv')
    for csvFile in newCsvList:
        if csvFile not in csvList:
            time.sleep(10)# to allow csv to be written by pupa imaging script completley
            print('\n'+csvDir+csvFile+' at '+present_time())
            csvData = np.genfromtxt(csvDir+csvFile, delimiter=',', dtype = 'int16')
            data = csvToData(csvData, step, csvFile.rstrip('.csv')+'_eucdisAngles.txt')
            plotDistance(data, csvFile.rstrip('.csv'), csvFile.rstrip('.csv')+'_eucDis.png')
            try:
                imFolderList = os.listdir(imDataDir)
                for dirs in imFolderList:
                    if len(os.listdir(imDataDir+dirs))>=119900:
                        time.sleep(2)
                        print("\nArchiving "+ dirs+" on: "+present_time())
                        createTar(dirs, imDataDir, outDir)
                        print("Archive created on: "+present_time())
            except:
                pass
        else:
            pass
    csvList = newCsvList
    time.sleep(2)
    sys.stdout.write("\rWaiting for next file")
    sys.stdout.flush()


#--------------------------------------------------------------------------


'''

        if len(os.listdir(inDir+dirs))>=119900:
            time.sleep(2)
            print "\nArchiving "+ dirs+" on: "+present_time()
            subprocess.call(['tar', 'rf',outDir+dirs+'.tar',\
                            inDir+dirs,'--totals', '--remove-files'])
            print "Archive created on: "+present_time()


'''




















