#!/bin/bash
fName='/media/data_ssd/tmp/imageData_1/20161012_173506_6.tar'
fName='/media/fly/ncbsStorage/twitchData/w1118/20150518_2/imageData/20150519_2_032308.tar'
#fName='/media/fly/ncbsStorage/twitchData/w1118/20150326_20150518/imageData/20150519_2_032308.tar'
#date; tar -tf $fName | wc -l; date
date; 7z l $fName | wc -l; date


#date; ffmpeg -v 5 -i "/media/data_ssd/rawMovies/CS/20160926_CS_data/imageData/20160926_234709.avi" -f null - 2>&1; date

#file='/media/data_ssd/rawMovies/OK371_90-101hr/20170415_234533_OK371xTrpA1_90-101hr/imageData/20170416_133415.avi'
#date; ffmpeg -v 5 -i $file -f null - 2>&1; date



