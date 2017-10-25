#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:13:21 2017

@author: pointgrey
"""
"""
Created on Sun Oct 11 02:10:20 2015

@author: pointgrey
"""
import serial as sr
from datetime import datetime
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import sys


print 'Enter the pupa collection details :<date_time (20170410_2200)>'
pupaDetails = raw_input()

print 'Enter APF for shift to 32C'
apfTo32 = raw_input()

print 'Enter hours to keep at 32C'
duration32 = raw_input()


collDetails = datetime.strptime(pupaDetails, '%Y%m%d_%H%M')
shiftStartHour = datetime.fromtimestamp((int(collDetails.strftime('%s'))\
                        +(int(apfTo32)*3600)))
toShiftStartHour = float((shiftStartHour-datetime.now()).total_seconds())/3600

tempVariations = np.array([[23,20.1],
                           [32,11],
                           [23,100],
                           [32,6],
                           [23,4],
                           [32,6],
                           [23,4],
                           [32,6],
                           [23,4],
                           [32,6],
                           [23,4],
                           [32,6],
                           [23,24]],dtype='float32')
tempVariations[0][1] = toShiftStartHour
tempVariations[1][1] = duration32

print tempVariations
lidValue = 0
lidHighValue = 70
lidLowValue = 5
try:
    tempVariations = np.array([[int(sys.argv[1]), 100]],dtype='float32')
except:
    pass


def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d;%H%M%S')
def present_time1():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def logFileWrite(content):
    '''Create log files'''
    try:
        logFile = open(logFileName,'a')# Trying to create a new file or open one
        logFile.write('\n')
        logFile.write(content)
        logFile.close
    except:
        print('Something went wrong! Can\'t create log file')

def getLedState():
    t = int(datetime.now().strftime('%H%M%S'))
    if (t<=80000):
        ledState = '0'
    elif (80000<t<=200000):
        ledState = '1'
    elif (200000<=t):
        ledState = '0'
    return ledState

def getSerData(ser):
    serData = ser.readline().lstrip('\r').rstrip("\nr")
    logFileWrite(serData+';'+present_time())
    serialData = serData.split(';')
    try:
        sys.stdout.write(clearSys+"SetTemp: %s, CurrTemp: %s, ChamberHV:%s, ardLidHV:%s at: %s\n"%(\
            serialData[4],serialData[0],serialData[1],serialData[2],present_time1() ))
        return int(serialData[4]),float(serialData[0]),int(serialData[2])
    except:
        return 0,0,0

def runTempProfile(tempVariations, ser):
    setTemp = tempVariations[0,0]
    ledState = '0'
    tick = datetime.now()
    tock = datetime.now()
    arduinoSetTemp = 0
    arduinoCurrTemp = 0
    arduinolidHV = 0
    for tempVariation in range(0, len(tempVariations)):
        lidHV = lidValue
        lastTempChange = datetime.now()
        nextTvar = datetime.fromtimestamp((int(lastTempChange.strftime('%s')))+\
                                            (tempVariations[tempVariation,1]*3600))
        while (tock-tick).total_seconds()<(np.sum(tempVariations[0:tempVariation+1,1])*3600):
            setTemp_new = tempVariations[tempVariation,0]
            ledState_new = getLedState()
            tLeft = (tempVariations[tempVariation,1])-\
                            ((tock-lastTempChange).total_seconds()/3600)
            if setTemp != setTemp_new or ledState_new != ledState:
                print ledState_new, setTemp_new, tock, tick #print these values at any temp change
                for pr in range(0,60):
                    print  '\n'
                setTemp = copy.copy(setTemp_new)
                ledState = copy.copy(ledState_new)
                print 'ERROR-----setTemp != setTemp_new or ledState_new != ledState-----ERROR'
#                print '1::'+ledState_new +','+ str(setTemp_new)+','+str(lidHV)
                ser.write('\r'+str(int(setTemp_new)))
#                time.sleep(2)
            arduinoSetTemp, arduinoCurrTemp, arduinolidHV = getSerData(ser)
            if arduinoSetTemp != setTemp:
                print 'ERROR-----arduinoSetTemp != setTemp-----ERROR'
#                print '2::'+ledState_new +','+ str(setTemp)+','+str(lidHV)
                ser.write('\r'+str(int(setTemp)))
#                time.sleep(2)
            if float(arduinoCurrTemp)< (setTemp-0.25):
                if setTemp>25:
                    lidHV = lidHighValue
                else:
                    lidHV = lidLowValue
                print 'ERROR-----arduinoCurrTemp< (setTemp-0.25) and arduinolidHV<lidHV-----ERROR'
#                print '3::'+ledState_new +','+ str(setTemp)+','+str(lidHV)
                ser.write('\r'+str(int(setTemp)))
#                time.sleep(2)
            if float(arduinoCurrTemp)>= (setTemp+0.25) and arduinolidHV != 0:
                lidHV = 0
                print 'ERROR-----arduinoCurrTemp >= (setTemp+0.25) and arduinolidHV != 0-----ERROR'
#                print '4::'+ledState_new +','+ str(setTemp)+','+str(lidHV)
                ser.write('\r'+str(int(setTemp)))
#                time.sleep(2)
            sys.stdout.write("TempProfile step: %d, Total Duration: %d hours, Set Temp: %d, \
LED State: %s\n"%(tempVariation,tempVariations[tempVariation,1], setTemp_new, ledState_new))
            sys.stdout.flush()
            sys.stdout.write("Next temp change at %s after %.2f hours\r"%(nextTvar, tLeft))
            sys.stdout.flush()
            tock = datetime.now()
#               time.sleep(1)


logFileName = '/home/pointgrey/ArduinoLogs/arduinolog_'+present_time1()+'.csv'
ledState = '0'
ser = sr.Serial(port='/dev/ttyACM0',baudrate=9600,timeout=3)
time.sleep(2)
setTemp =tempVariations[0,0]
sr.Serial()
print  '\n'+str(tempVariations)
logFileWrite(str(tempVariations))
print "Started temp profile on: "+present_time()
print 'Starting LedState = %s, and temperature = %s' %(ledState, setTemp)
ser.write('\r'+str(int(setTemp)))


CURSOR_UP_TWO = '\x1b[1A' # for moving sys.stdout cursor up by one line
ERASE_LINE = '\x1b[2K' # for clearing sys.stdout data from the line
clearSys =  CURSOR_UP_TWO+ERASE_LINE

try:
    runTempProfile(tempVariations, ser)
except KeyboardInterrupt:
    sys.exit        






#
'''
http://stackoverflow.com/questions/39177788/python-sys-stdout-flush-on-2-lines-in-python-2-7

CURSOR_UP_ONE = '\x1b[2A' 
ERASE_LINE = '\x1b[2K'

data_on_first_line = CURSOR_UP_ONE + ERASE_LINE + "abc\n"
sys.stdout.write(data_on_first_line)

data_on_second_line = "def\r"
sys.stdout.write(data_on_second_line)
sys.stdout.flush()



try:
    keep looking for the folder, if no folder present: pass 
    else
    get Camploop.txt
except:
    print 'pupa age: unknown, no camloop present'

sys.stdout.write("\r%s: %d"%(present_time(),nFrame))
sys.stdout.flush()


'''



















































