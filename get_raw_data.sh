#!/bin/bash

mkdir raw_data
cd raw_data

wget https://cgl.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
wget https://web.engr.oregonstate.edu/~lif/SegTrack2/SegTrackv2.zip
wget https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Trainingset.zip
wget https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Testset.zip

unzip DAVIS-data.zip
unzip SegTrackv2.zip
unzip FBMS_Trainingset.zip
unzip FBMS_Testset.zip

mkdir FBMS59

mv Trainingset FBMS59/
mv Testset FBMS59/

rm SegTrackv2/JPEGImages/CSI_NY.jpg
