import cv2
import numpy as np
import yaml
import os
import sys

#  the directory of imageDataSet and descriptorsSet
imgDir = 'E://WorkSpace//Python//VLAD//images//'
desDir = 'E://WorkSpace//Python//VLAD//descriptors//'

sift = cv2.xfeatures2d.SIFT_create()

Set = None
dirlist = os.listdir(imgDir)
total = len(dirlist)
index=0
for imgName in dirlist:
	index+=1
	percent = index/total*100
	print('SIFT Extraction Progress: %6.2f%%\timage: %s'%(percent,imgName),end='\r')#console

	Input = cv2.imread(imgDir+imgName,cv2.IMREAD_GRAYSCALE)
	keypoints, descriptors = sift.detectAndCompute(Input,None)
	if descriptors is not None:
		print(len(descriptors))
	if index ==1:
		Set = descriptors
	else:
		Set = np.append(Set,descriptors,axis=0)
	
	# desFileName = imgName.replace('jpg','yaml')
	# output = open(desDir+desFileName, 'w')
	# data = {}
	# data['image'] = imgName
	# data['descriptors'] = descriptors
	# yaml.dump(data, output) 
	# output.close()
print('SIFT Extraction Done!')

