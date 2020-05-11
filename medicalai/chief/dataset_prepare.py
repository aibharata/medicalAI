#    Copyright 2020-2022 AIBharata Emerging Technologies Pvt. Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import absolute_import
from  medicalai.__about__ import __version__
import os
from  .uFuncs import timeit
import pydicom as dicomProcessor
import pandas as pd
def datasetFolderStructureValidate(folder):
	dirs = os.listdir(folder)
	test = train = val = False
	dirsDict = {}
	if ('test' in dirs):
		print("'test' Folder Present")
		test = True 
	if ('train' in dirs):
		print("'train' Folder Present")
		train = True
	if ('validate' in dirs):
		print("'validate' Folder Present")

	assert train == True, "NO 'train' Folder found"
	assert test == True, "NO 'test' Folder found"
	
	return dirs

def getLabelsFromFolder(folder):
	dirs  = datasetFolderStructureValidate(folder)
	labels = os.listdir(os.path.join(folder, 'train'))
	labels_test = labels_val = None
	if ('test' in dirs):
		labels_test = os.listdir(os.path.join(folder, 'test'))
	if ('validate' in dirs):
		labels_val = os.listdir(os.path.join(folder, 'validate'))
	if labels_test:
		assert labels ==  labels_test, "Train and Test Labels Dont Match"
	if labels_val:
		assert labels ==  labels_val, "Train and Validation Labels Dont Match"
	
	#dirs = [os.path.join(folder,x) for x in dirs]
	return dirs, labels

import numpy as np
from PIL import Image
import tqdm
from sklearn.preprocessing import LabelBinarizer,LabelEncoder

def _labelMapper(labelNames, labelMap):
	y_idx = []
	for sample in labelNames:
		y_idx.append(np.array([labelMap[sample]]))
	return y_idx


def _rgb_dataset_from_folder(Folder, labels, labelMap, inpPreProc):
	DataSet = None
	LabelSet =[]
	for label in labels:
		classFolder = os.path.join(Folder, label)
		SetItems = os.listdir(classFolder)
		for image in SetItems:
			LabelSet.append(label)
		if DataSet is not None:
			DataSet = np.vstack((DataSet,inpPreProc.resizeDataSetfromFolder(classFolder)))
		else:
			DataSet = inpPreProc.resizeDataSetfromFolder(classFolder)
	y_train = _labelMapper(LabelSet, labelMap)
	y_train = np.array(y_train)
	return (DataSet, y_train)

def _baseLabelMapper(labels):
	le = LabelEncoder()
	intEncoded = le.fit_transform(labels)
	labelDict = {}
	for k,v in zip(labels,intEncoded):
		labelDict[k] = v
	return labelDict



class INPUT_PROCESSOR:
	def __init__(self, targetDim=(31,31), samplingMethod=None,normalize=False, color_mode='RGB', rescale=None , dtype='float32'):
			
		self.output_size		= targetDim
		self.normalize			= normalize
		self.rescale 			= rescale
		self.dtype 				= dtype
		self.color_mode			= color_mode
		self.samplingMethodName = 'nearest' if samplingMethod==None else samplingMethod
		if samplingMethod: 
			if samplingMethod.lower()=='box':
				self.samplingMethod =Image.BOX
			elif samplingMethod.lower()=='nearest':
				self.samplingMethod =Image.NEAREST
			elif samplingMethod.lower()=='bilinear':
				self.samplingMethod =Image.BILINEAR
			elif samplingMethod.lower()=='hamming':
				self.samplingMethod =Image.HAMMING
			elif samplingMethod.lower()=='bicubic':
				self.samplingMethod =Image.BICUBIC
			elif samplingMethod.lower()=='lanczos':
				self.samplingMethod =Image.LANCZOS		
			else:
				self.samplingMethod =Image.NEAREST
		else:
			self.samplingMethod =Image.NEAREST

	def resizeDataSet(self,dataset):
		processedData = []
		for i in tqdm.tqdm(range(0,dataset.shape[0])):
			im = Image.fromarray(dataset[i])
			np_im = self._process_single_image(im)
				
			processedData.append(np_im)
		processedData = np.array(processedData)
		if self.normalize:
			processedData = processedData*(1./255)
		if self.rescale is not None:
			processedData = processedData*self.rescale
		return processedData

	def _process_single_image(self, img):
		if self.color_mode.upper() == 'RGB':
			if img.mode != 'RGB':
				img = img.convert('RGB')
		elif self.color_mode.upper() == 'RGBA':
			if img.mode != 'RGBA':
				img = img.convert('RGBA')
		elif color_mode.upper() == 'GRAYSCALE':
			if img.mode != 'L':
				img = img.convert('L')
		img = img.resize((self.output_size[0:2]),self.samplingMethod)
		np_im = np.array(img, self.dtype)
		return np_im

	def resizeDataSetfromFolder(self,folder):
		processedData = []
		imageNames = os.listdir(folder)
		for i in tqdm.tqdm(range(0,len(imageNames))):
			img = Image.open(os.path.join(folder, imageNames[i]))
			np_im = self._process_single_image(img)
			processedData.append(np_im)
		processedData = np.array(processedData)
		if self.normalize:
			processedData = processedData*(1./255)
		if self.rescale is not None:
			processedData = processedData*self.rescale
		return processedData


	def processImage(self,image):
		if isinstance(image,np.ndarray):
			if len(image.shape)>3:
				image = np.squeeze(image,0)
			img = Image.fromarray(np.uint8((image)*255))
		else:
			try:
				sep = os.path.splitext(image)
				fileExt = sep[-1]
				fileName = sep[0]
			except: 
				sep = os.path.splitext(image.path)
				fileExt = sep[-1]
				fileName = sep[0]
			if fileExt== '.dcm':
				ds = dicomProcessor.dcmread(image)
				img = ds.pixel_array
				im = Image.fromarray(img)
				fName = fileName+".png"
				im.save(fName)
				img = Image.open(fName)
			else:
				img = Image.open(image)
		np_im = self._process_single_image(img)
		if self.normalize:
			np_im = np_im*(1./255)
		if self.rescale is not None:
			np_im = np_im*self.rescale
		np_im = np.expand_dims(np_im, axis=0)
		return np_im



class InputProcessorFromMeta(INPUT_PROCESSOR):
	def __init__(self, metaFile):
		params = metaLoader(metaFile)
		super().__init__(targetDim= params['network_input_dim'], samplingMethod=params['samplingMethodName'], normalize=params['normalize'],rescale=params['rescale'], )
		self.classes = params['labels']
		self.labels = params['labels']
		self.labelMap = params['labelMap'] 
		

class datasetManager(INPUT_PROCESSOR):
	def __init__(self,folder, targetDim=(31,31), normalize=False , name = None, useCache=True , forceCleanCache = False):
		super().__init__(targetDim=targetDim, normalize=normalize)
		self.folder = folder
		if name ==None:
			self.name = folder.split('/')[-1].split('\\')[-1]
		else:
			self.name = name
		self.useCache = useCache
		self.dirs, self.labels = getLabelsFromFolder(folder)
		self.labelMap  = _baseLabelMapper(self.labels)
		self.cachefile = self.name+'_'+ str(self.output_size[0]) + '_' + str(self.output_size[1])+ '_'+str(self.samplingMethod)+'_N'+str(self.normalize)+'.npz'
		if self.useCache and not forceCleanCache and os.path.exists(self.cachefile):
			self.reload_data()
		else:
			self.process_dataset()

	@timeit	
	def convert_dataset(self):
			trainFolder = os.path.join(self.folder,'train')
			(self.x_train, self.y_train) = _rgb_dataset_from_folder(trainFolder, self.labels, self.labelMap, self)
			# Test Data Preparataio, if Present
			if ('test' in self.dirs):
				(self.x_test, self.y_test) = _rgb_dataset_from_folder(os.path.join(self.folder,'test'), self.labels, self.labelMap, self)
				print('Test Shape:', self.x_test.shape) 
				
			# Validation Data Preparataio, if Present
			if ('validate' in self.dirs):
				(self.x_val, self.y_val) = _rgb_dataset_from_folder(os.path.join(self.folder,'validate'), self.labels, self.labelMap, self)
				print('Validate Shape:', self.x_val.shape) 
				
			print('Train Shape:', self.x_train.shape)

	def process_dataset(self):
			self.convert_dataset()
			if self.useCache:
				self.compress_and_cache_data()
	@timeit
	def reload_data(self):
		print('Reloading Dataset from Cache')
		cached = np.load(self.cachefile, mmap_mode='r')
		(self.x_test, self.y_test) = (cached['x_test'], cached['y_test'])
		(self.x_train, self.y_train) = (cached['x_train'], cached['y_train'])
	
	@timeit
	def compress_and_cache_data(self):
		print('Caching Dataset')
		np.savez_compressed(self.cachefile, x_train=self.x_train, y_train=self.y_train,
									x_test = self.x_test, y_test = self.y_test)


	def load_data(self):
		return (self.x_train, self.y_train), (self.x_test, self.y_test)

class myDict(dict):
    pass

class datasetFromFolder(datasetManager):
	'''
	TODO: Fix samplingMethodName assignment
	'''
	def __init__(self,folder, targetDim=(31,31), normalize=False , name = None, useCache=True , forceCleanCache = False):
	 super().__init__(folder, targetDim, normalize, name, useCache, forceCleanCache)
	 self.train = myDict()
	 self.test = myDict()
	 (self.train.data, self.train.labels), (self.test.data, self.test.labels) = self.load_data()
	 self.load_dataset()
	 self.train.network_input_dim= targetDim
	 self.train.normalize= normalize
	 self.train.labelMap = self.labelMap
	 self.train.labelNames = self.labels
	 self.test.labelMap = self.labelMap
	 self.test.labelNames = self.labels

	 self.train.samplingMethodName = self.samplingMethodName
	
	def load_dataset(self): 
	 return self.train, self.test, self.labels

def datasetManagerFunc(folder,targetDim=(31,31), normalize=False):
	dirs, labels = getLabelsFromFolder(folder)
	labelMap  = _baseLabelMapper(labels)
	print(labelMap)

	# Create Preprocessor
	inpPreProc = INPUT_PROCESSOR(targetDim=targetDim, normalize=normalize)

	# Train Data Preparataion
	trainFolder = os.path.join(folder,'train')
	(x_train, y_train) = _rgb_dataset_from_folder(trainFolder, labels, labelMap, inpPreProc)

	# Test Data Preparataio, if Present
	if ('test' in dirs):
		(x_test, y_test) = _rgb_dataset_from_folder(os.path.join(folder,'test'), labels, labelMap, inpPreProc)
		print('Test Shape:', x_test.shape) 
		
	# Validation Data Preparataio, if Present
	if ('validate' in dirs):
		(x_val, y_val) = _rgb_dataset_from_folder(os.path.join(folder,'validate'), labels, labelMap, inpPreProc)
		print('Validate Shape:', x_val.shape) 
		
	print('Train Shape:', x_train.shape) 
	return (x_train, y_train), (x_test, y_test)



def convertlist2tuple(lst): 
	return tuple(lst) 

def metaLoader(metaFile):
    import json
    if '.json' not in metaFile:
        metaFile = metaFile+"_meta.json"
    with open(metaFile) as f:
        json_data = json.load(f)
    version = __version__
    if version != json_data['medicalai_version']:
        print('Meta File Generated using Medicalai v{:}.'.format(json_data['medicalai_version']))
        print('Current Installed version of Medicalai v{:}'.format(version))
        print('If any problem occurs during training/inference. Check medicalai page: {}'.format('https://github.com/aibharata/medicalAI'))
    json_data['config']['network_input_dim']= convertlist2tuple(json_data['config']['network_input_dim'])
    return json_data['config']


def metaSaver(labelMap, labels, normalize=None, rescale=None, network_input_dim=None, samplingMethodName=None, outputName=None):
	import json
	from collections import OrderedDict
	meta = OrderedDict()
	meta['medicalai_version'] = __version__
	meta['config'] = OrderedDict()
	for k,v in labelMap.items():
		labelMap[k] = int(v)
	meta['config']['labelMap'] = labelMap
	meta['config']['normalize'] = normalize
	meta['config']['rescale'] = rescale
	meta['config']['labels'] = labels
	meta['config']['network_input_dim'] = network_input_dim
	meta['config']['samplingMethodName'] = samplingMethodName
	
	with open(outputName+"_meta.json", "w") as f:
		json.dump(meta,f, sort_keys=True, indent=4)

import tensorflow as tf
class AUGMENTATION(object):
	def __init__(self, rotation_range = 12, fill_mode='constant', width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=False, vertical_flip=False,
				 brightness_range = (0.9, 1.1), zoom_range=(0.85, 1.15), rescale= 1./255,
				 shear_range = 0, channel_shift_range = 0, samplewise_center = False, samplewise_std_normalization = False,
				featurewise_center=False, featurewise_std_normalization=False,cval=0,
				preprocessing_function = None):
		self.rotation_range = rotation_range
		self.fill_mode = fill_mode
		self.width_shift_range = width_shift_range
		self.height_shift_range = height_shift_range
		self.horizontal_flip = horizontal_flip
		self.vertical_flip = vertical_flip
		self.brightness_range = brightness_range
		self.zoom_range = zoom_range
		self.rescale = rescale
		self.featurewise_center = featurewise_center
		self.featurewise_std_normalization = featurewise_std_normalization
		self.cval = cval
		self.shear_range = shear_range
		self.channel_shift_range = channel_shift_range
		self.samplewise_std_normalization = samplewise_std_normalization
		self.samplewise_center = samplewise_center
		self.preprocessing_function = preprocessing_function
		self.create_aug()

	def create_aug(self):
		self.trainAug = tf.keras.preprocessing.image.ImageDataGenerator(
						rotation_range = self.rotation_range,
						fill_mode = self.fill_mode,
						width_shift_range = self.width_shift_range,
						height_shift_range = self.height_shift_range,
						horizontal_flip = self.horizontal_flip,
						vertical_flip = self.vertical_flip,
						brightness_range = self.brightness_range,
						zoom_range = self.zoom_range,
						rescale = self.rescale,
						featurewise_center = self.featurewise_center,
						featurewise_std_normalization = self.featurewise_std_normalization,
						cval = self.cval,
						shear_range = self.shear_range,
						channel_shift_range = self.channel_shift_range,
						samplewise_std_normalization = self.samplewise_std_normalization,
						samplewise_center = self.samplewise_center,
						preprocessing_function = self.preprocessing_function,
                        )
		self.testAug = tf.keras.preprocessing.image.ImageDataGenerator(
                                            rescale=self.rescale
                                             )
		#return trainAug, testAug

from tensorflow.keras.utils import Sequence,to_categorical
class medicalai_generator(tf.keras.preprocessing.image.ImageDataGenerator):
	def __init__(self):
		super().__init__()
		self.one_hot_labels = to_categorical(self.labels)

def safe_labelmap_converter(labelMap):
	labs = [0 for x in list(labelMap.keys())]
	for k,v in labelMap.items():
		labs[v]=k
	return labs

class datasetGenFromFolder(object):
	"""
	folder : The directory must be set to the path where your `n` classes of folders are present.
	targetDim : The target_size is the size of your input images to the neural network.
	class_mode : Set `binary` if classifying only two classes, if not set to `categorical`, in case of an Autoencoder system, both input and the output would probably be the same image, for this case set to `input`.
	color_mode: `grayscale` for black and white or grayscale, `rgb` for three color channels.
	batch_size: Number of images to be yielded from the generator per batch. If training fails lower this number.
	"""
	def __init__(self, folder, targetDim=(224,224), normalize=False , batch_size = 16, augmentation = True, 
					color_mode="rgb",class_mode="sparse",shuffle=True, seed=17
					):

		self.folder = folder
		self.targetDim = targetDim
		self.normalize = normalize
		self.batch_size = batch_size
		
		self.color_mode = color_mode
		self.class_mode = class_mode
		self.seed = seed
		self.shuffle = shuffle
		self.trainGen = myDict()
		self.testGen = myDict()
		self.class_weights = myDict()

		if isinstance(augmentation, AUGMENTATION):
			self.augmentation = augmentation
		else:
			if augmentation==True or augmentation=='True' or augmentation=='Default':
				self.augmentation  = AUGMENTATION()
			else:
				self.augmentation = myDict()
				self.augmentation.trainAug = tf.keras.preprocessing.image.ImageDataGenerator()
				self.augmentation.testAug = tf.keras.preprocessing.image.ImageDataGenerator()

		self.trainGen.generator = self.augmentation.trainAug.flow_from_directory(
				directory=os.path.join(self.folder,"train"),
				target_size=targetDim,
				batch_size=self.batch_size,
				color_mode = self.color_mode,
				class_mode = self.class_mode,
				seed = self.seed,
				shuffle = self.shuffle,
				)

		self.testGen.generator = self.augmentation.testAug.flow_from_directory(
				directory=os.path.join(self.folder,"test"),
				target_size=targetDim,
				batch_size=self.batch_size, #1
				color_mode = self.color_mode,
				class_mode = self.class_mode,
				seed = self.seed,
				shuffle = False,
				)				
		self.trainGen.STEP_SIZE= np.ceil(self.trainGen.generator.n//self.trainGen.generator.batch_size)
		self.testGen.STEP_SIZE= np.ceil(self.testGen.generator.n//self.testGen.generator.batch_size)
		self.labelMap = self.trainGen.generator.class_indices
		self.trainGen.labelMap = self.trainGen.generator.class_indices
		self.testGen.labelMap = self.trainGen.generator.class_indices
		self.trainGen.labelNames = safe_labelmap_converter(self.trainGen.labelMap)
		self.testGen.labelNames = safe_labelmap_converter(self.testGen.labelMap)
		if len(self.trainGen.generator.labels.shape)==1:
			self.trainGen.generator.one_hot_labels = to_categorical(self.trainGen.generator.labels)
			self.testGen.generator.one_hot_labels = to_categorical(self.testGen.generator.labels)
		else:
			self.trainGen.generator.one_hot_labels = self.trainGen.generator.labels
			self.testGen.generator.one_hot_labels = self.testGen.generator.labels

	def load_generator(self):
		return self.trainGen, self.testGen

	def get_class_weights(self):
		self.pos = np.sum(np.array(self.trainGen.generator.one_hot_labels)==1, axis=0)
		for i in range(len(self.pos)):
			self.class_weights[i]=(np.sum(self.pos))/(len(self.pos)*self.pos[i])
		return self.class_weights
	
	def get_numpy(self, generator):
		prevBSize =generator.generator.batch_size
		generator.generator.batch_size  = generator.generator.samples
		dataset_as_tuple = next(generator.generator)
		generator.generator.batch_size = prevBSize
		return dataset_as_tuple

#############################################################################################
#Data from data frame implementation
class datasetGenFromDataframe(object):
	"""Creates Keras Dataset Generator for Handling Large Datasets from DataFrame.
	
	Arguments:
		csv_path: folder containing train.csv and test.csv.
		folder: The directory must be set to the path where your training images are present.
		x_col: Name of column containing image name, `default = name`.
		y_col: Name of column for labels, `default = labels`.
		targetDim: The target_size is the size of your input images to the neural network.
		class_mode: Set `binary` if classifying only two classes, if not set to `categorical`, in case of an Autoencoder system, both input and the output would probably be the same image, for this case set to `input`.
		color_mode: `grayscale` for black and white or grayscale, `rgb` for three color channels.
		batch_size: Number of images to be yielded from the generator per batch. If training fails lower this number.
	"""

	def __init__(self, folder, csv_path='.' , x_col = "name", y_col = "labels", targetDim=(224,224), normalize=False , batch_size = 16, augmentation = True, 
					color_mode="rgb", class_mode="sparse", shuffle=True, seed=17):
		self.csv_path = csv_path
		self.folder = folder
		self.x_col = x_col
		self.y_col = y_col
		self.targetDim = targetDim
		self.normalize = normalize
		self.batch_size = batch_size
		
		self.color_mode = color_mode
		self.class_mode = class_mode
		self.seed = seed
		self.shuffle = shuffle
		self.trainGen = myDict()
		self.testGen = myDict()
		self.class_weights = myDict()

		if isinstance(augmentation, AUGMENTATION):
			self.augmentation = augmentation
		else:
			if augmentation==True or augmentation=='True' or augmentation=='Default':
				self.augmentation  = AUGMENTATION()
			else:
				self.augmentation = myDict()
				self.augmentation.trainAug = tf.keras.preprocessing.image.ImageDataGenerator()
				self.augmentation.testAug = tf.keras.preprocessing.image.ImageDataGenerator()

		self.traindf = pd.read_csv(os.path.join(csv_path,"train.csv"),dtype=str)
		self.testdf = pd.read_csv(os.path.join(csv_path,"test.csv"),dtype=str)
		print("[INFO]: Succesfully read train.csv & test.csv")
		print("[INFO]: Gathering images for generator")
		self.trainGen.generator = self.augmentation.trainAug.flow_from_dataframe(
				dataframe = self.traindf,
				directory=os.path.join(self.folder,"train"),
				x_col = self.x_col,
				y_col = self.y_col,
				target_size=targetDim,
				batch_size=self.batch_size,
				color_mode = self.color_mode,
				class_mode = self.class_mode,
				seed = self.seed,
				shuffle = self.shuffle,
				validate_filenames = False
				)

		self.testGen.generator = self.augmentation.testAug.flow_from_dataframe(
				dataframe = self.testdf,
				directory=os.path.join(self.folder,"test"),
				x_col = self.x_col,
				y_col = self.y_col,
				target_size=targetDim,
				batch_size=self.batch_size,
				color_mode = self.color_mode,
				class_mode = self.class_mode,
				seed = self.seed,
				validate_filenames = False,
				shuffle = False
				)				
		self.trainGen.STEP_SIZE= np.ceil(self.trainGen.generator.n//self.trainGen.generator.batch_size)
		self.testGen.STEP_SIZE= np.ceil(self.testGen.generator.n//self.testGen.generator.batch_size)
		self.labelMap = self.trainGen.generator.class_indices
		self.trainGen.labelMap = self.trainGen.generator.class_indices
		self.testGen.labelMap = self.trainGen.generator.class_indices
		self.trainGen.labelNames = safe_labelmap_converter(self.trainGen.labelMap)
		self.testGen.labelNames = safe_labelmap_converter(self.testGen.labelMap)
		if len(np.asarray(self.trainGen.generator.labels).shape)==1:
			print("[INFO]: Converting labes to one_hot_labels")
			self.trainGen.generator.one_hot_labels = to_categorical(self.trainGen.generator.labels)
			self.testGen.generator.one_hot_labels = to_categorical(self.testGen.generator.labels)
		else:
			self.trainGen.generator.one_hot_labels = self.trainGen.generator.labels
			self.testGen.generator.one_hot_labels = self.testGen.generator.labels

	def load_generator(self):
		return self.trainGen, self.testGen

	def get_class_weights(self):
		self.pos = np.sum(np.array(self.trainGen.generator.one_hot_labels)==1, axis=0)
		for i in range(len(self.pos)):
			self.class_weights[i]=(np.sum(self.pos))/(len(self.pos)*self.pos[i])
		return self.class_weights
	
	def get_numpy(self, generator):
		prevBSize =generator.generator.batch_size
		generator.generator.batch_size  = generator.generator.samples
		dataset_as_tuple = next(generator.generator)
		generator.generator.batch_size = prevBSize
		return dataset_as_tuple

#############################################################################################

if __name__ == "__main__":
	mainFolder = "chest-xray-pnumonia-covid19"
	v = datasetManager(mainFolder,targetDim=(96,96), normalize=False)	 

