import cv2
import numpy as np
import yaml
import os
import sys
import pandas as pd
from sklearn import preprocessing
import gc

class VLAD(object):
	"""
	docstring for VLAD
	the format of argument as follows
		Directory['images'] = imgDir
		Directory['results'] = resultDir
	"""

	def __init__(self, Directory):
		self.imgDir = Directory['images']
		self.resultsDir = Directory['results']
		self.imgList = os.listdir(self.imgDir)

		self.descriptorsIndex = None
		self.dataSet = None
		self.label = None
		self.center = None
		self.DataFrame =None
		self.VALDs = None

	def SIFTExtract(self):
		'''
		SIFTExtract
		'''

		print('SIFT Extraction Start!')
		dirlist = self.imgList
		total = len(dirlist)
		index=0
		sift = cv2.xfeatures2d.SIFT_create()

		featuresUnclustered=None
		featureIndex = []
		for imgName in dirlist:
			index+=1
			percent = index/total*100
			print('SIFT Extraction Progress: %6.2f%%\timage: %s'%(percent,imgName),end='\r')#console
			### SIFT feature compute
			Input = cv2.imread(imgDir+imgName,cv2.IMREAD_GRAYSCALE)
			keypoints, descriptors = sift.detectAndCompute(Input,None)

			if descriptors is not None:
				if featuresUnclustered is None:
					featuresUnclustered = descriptors
				else:
					featuresUnclustered = np.append(featuresUnclustered,descriptors,axis=0)
				featureIndex.extend([imgName]*len(descriptors))
				# featuresUnclustered.append(descriptors)
		print('SIFT Extraction Done!')

		self.dataSet = featuresUnclustered
		self.descriptorsIndex = featureIndex
	

	def Kmeans(self):
		print('Kmeans procedure Start!')
		
		### dataSet  for Kmeans
		dataSet = self.dataSet
		### kmeans
		print('kmeans clusting...')
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		#						samples,nclusters(K),bestLabels,criteria,attempts,flags
		ret,label,center=cv2.kmeans(dataSet,16,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
		print('Kmeans procedure Finished!')

		self.label = label
		self.center = center
		self.DataProcess()

	def DataProcess(self):
		imgID = self.descriptorsIndex
		label = self.label
		dataSet = self.dataSet
		center = self.center
		resultsDir = self.resultsDir

		c1 = ['label']
		c2 = ['value']
		for x in range(128):
			c1.append('descriptor')
			c2.append(x)
		col_list=[c1,c2]

		col_idx = pd.MultiIndex.from_arrays(col_list,names=['class1','class2'])
		idx = pd.Index(imgID,name='ImageName')

		dfData = np.column_stack((label,dataSet))
		df = pd.DataFrame(dfData ,columns=col_idx,index=idx)
		self.DataFrame =df 
		### 
		store = pd.HDFStore(resultsDir+'labeledDataSet.h5')
		store.put('labelData',df,format='table',append=False)
		# df.to_hdf(store,'table',append=False)
		store.close()
		### 

		centerFile = 'center.yaml'
		output = open(resultsDir+centerFile, 'w')
		data = {}
		data['center'] = center
		yaml.dump(data, output) 
		output.close()

		### garbage collection|release memory of following objects
		del self.dataSet
		del self.label
		del self.descriptorsIndex
		gc.collect()

	def VLADgen(self):
		df = self.DataFrame
		center = self.center
		imgList = self.imgList 

		VLADs = []
		total = len(imgList)
		index = 0
		for image in imgList:
			### show progress
			index+=1
			percent = index/total*100
			print('VLAD Generation Progress: %6.2f%%\timage: %s'%(percent,image),end='\r')#console
			
			eachImage = df.loc[image]
			descriptors = eachImage['descriptor'].values
			labels = eachImage['label'].values

			VLAD = np.zeros((16,128))
			for i in range(len(labels)):
				label = int(labels[i][0])
				c = center[label]
				d = descriptors[i]
				VLAD[label]+=d-c
			VLAD_normalized = preprocessing.normalize(VLAD.reshape((1,-1)), norm='l2')
			VLADs.append(VLAD_normalized)##List

		idx = pd.Index(imgList,name='ImageName')
		VLAD_dfData = np.vstack(VLADs)
		VLAD_df = pd.DataFrame(VLAD_dfData,index=idx)

		self.VALDs = VLAD_df
		#####
		#VLAD_df.iloc[i] get the i-st VLAD feature
		#VLAD_df.index[i] get the i-st VLAD index(image name)
		#####

	def resultProcess(self):
		resultsDir = self.resultsDir
		VLAD_df = self.VALDs
		### 
		store = pd.HDFStore(resultsDir+'VLADSet.h5')
		store.put('VLADSet',VLAD_df,format='fixed',append=False)
		#  the format is set 'fixed' without columns index
		store.close()

	def run(self):
		self.SIFTExtract()
		self.kmeans()
		self.VLADgen()
		self.resultProcess()

if __name__ == '__main__':
	Directory={}
	#  the directory of imageDataSet and descriptorsSet
	# imgDir = 'E://ResearchSpace//oxbuild_images//'
	imgDir = 'E://WorkSpace//Python//VLAD//images//'
	resultDir = 'E://WorkSpace//Python//VLAD//results//'

	Directory['images'] = imgDir
	Directory['results'] = resultDir

	vt = VLAD(Directory)
	vt.SIFTExtract()
	vt.Kmeans()
	vt.VLADgen()
	vt.resultProcess()

