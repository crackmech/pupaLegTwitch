import tarfile
import time

from datetime import datetime
import re

def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


fName='/media/fly/ncbsStorage/twitchData/w1118/20150518_2/imageData/20150519_2_032308.tar'

#tarNames = []
#with tarfile.open(fName, 'r') as tar_file:
#    print(tar_file.getnames())
#    tarNames.append(tar_file.getnames())
#print(len(tarNames))
#
#tar_file = tarfile.open(fName, 'r')
#print(tar_file.getnames())
#tarNames.append(tar_file.getnames())
#
#members = tar_file.getmembers()

def readTarContents(tarName, nCurThrds):
    '''
    read contents of the imageData tar folder into a dict
    '''
    readTime = time.time()
    tar = tarfile.open(tarName,'r|') 
    tarStack = []
    for f in tar:
        if f.isfile():
            #c = tar.extractfile(f).read()
            tarStack.append(f.get_info()['name'])
    tar.close()
    print('Read in: %.02f Seconds, # Current Threads: %d '%(\
            (time.time()-readTime), nCurThrds))
    return tarStack

print('Started at: %s'%present_time())
tarContents = readTarContents(fName, 4)
print(len(tarContents))
print('Ended at: %s'%present_time())

print('Started reading file names')
startTime = time.time()
tarNames = []
with tarfile.open(fName, 'r') as tar_file:
    tarNames.append(tar_file.getnames())
print(len(tarNames))
print('Ended reading file names in: %0.3f'%(time.time()-startTime))


print('Ended at: %s'%present_time())
