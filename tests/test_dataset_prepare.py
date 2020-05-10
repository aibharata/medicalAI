# test_dataset_prepare.py
import pytest
import medicalai as ai
import tensorflow as tf

import pandas as pd

def test_augmentation():
		v = ai.AUGMENTATION()
		print('AUG Type:', type(v.trainAug))
		assert isinstance(v.trainAug, tf.keras.preprocessing.image.ImageDataGenerator) , "augmentation: TrainAug - Default Initialization Failed"
		assert isinstance(v.testAug, tf.keras.preprocessing.image.ImageDataGenerator) , "augmentation: TestAug - Default Initialization Failed"

def test_datasetGenFromFolder():
	dsfolder = '/home/aditya/AIBH_Intern_09_Aditya/data'
	print(10*'-','Checking True or Default Augmentation',10*'-',)
	trainGen, testGen = ai.datasetGenFromFolder(dsfolder).load_generator()
	#print('DS GenType:', trainGen.STEP_SIZE, type(trainGen.STEP_SIZE) )
	assert isinstance(trainGen.STEP_SIZE, float), "datasetGenFromFolder: Failed"

	print(10*'-','Checking False or No Augmentation',10*'-',)
	trainGen, testGen = ai.datasetGenFromFolder(dsfolder, augmentation=False).load_generator()
	assert isinstance(trainGen.STEP_SIZE, float), "datasetGenFromFolder: Failed"

	print(10*'-','Checking Augmentation Passing',10*'-',)
	v = ai.AUGMENTATION()
	trainGen, testGen = ai.datasetGenFromFolder(dsfolder, augmentation=v).load_generator()
	assert isinstance(trainGen.STEP_SIZE, float), "datasetGenFromFolder: Failed"

def test_datasetGenFromDataframe():
	dsfolder = '/home/aditya/AIBH_Intern_09_Aditya/data'
	print(10*'-','Checking True or Default Augmentation',10*'-',)
	trainGen, testGen = ai.datasetGenFromDataframe(dsfolder).load_generator()
	#print('DS GenType:', trainGen.STEP_SIZE, type(trainGen.STEP_SIZE) )
	assert isinstance(trainGen.STEP_SIZE, float), "datasetGenFromDataframe: Failed"

	print(10*'-','Checking False or No Augmentation',10*'-',)
	trainGen, testGen = ai.datasetGenFromDataframe(dsfolder, augmentation=False).load_generator()
	assert isinstance(trainGen.STEP_SIZE, float), "datasetGenFromDataframe: Failed"

	print(10*'-','Checking Augmentation Passing',10*'-',)
	v = ai.AUGMENTATION()
	trainGen, testGen = ai.datasetGenFromDataframe(dsfolder, augmentation=v).load_generator()
	assert isinstance(trainGen.STEP_SIZE, float), "datasetGenFromDataframe: Failed"
