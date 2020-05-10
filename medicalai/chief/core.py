#	 Copyright 2020-2022 AIBharata Emerging Technologies Pvt. Ltd.

#	 Licensed under the Apache License, Version 2.0 (the "License");
#	 you may not use this file except in compliance with the License.
#	 You may obtain a copy of the License at

#		 http://www.apache.org/licenses/LICENSE-2.0

#	 Unless required by applicable law or agreed to in writing, software
#	 distributed under the License is distributed on an "AS IS" BASIS,
#	 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#	 See the License for the specific language governing permissions and
#	 limitations under the License.

from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import time
import tensorflow as tf
from tensorflow.keras.models import Sequential,model_from_json
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import models

import numpy as np
import matplotlib.pyplot as plt
from  . import networks 
from  . import dataset_prepare as dataprc 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report,cohen_kappa_score
from .model_metrics import *
from .xai import *
from .uFuncs import *

	
	
def create_model_output_folder(outputName):
	if "\\" in outputName:
		outputName=outputName.replace('\\','/')
	folder= outputName.replace(outputName.split('/')[-1],'')
	if not os.path.exists(folder):
		os.makedirs(folder)	

def check_model_exists(outputName):
	if not os.path.exists(outputName+'_arch.json'):
		print("\n\n\n \tModel Doesnt Exist \n\n\n")
		return False
	else:
		print("\n\n\n \tUsing Model: {:}.json \n\n\n".format(outputName))
		return True
		
def save_model_and_weights(model,outputName):
	create_model_output_folder(outputName)
	model.save_weights(outputName+'_wgts.h5')
	open(outputName+'_arch.json', 'w').write(model.to_json())


def load_model_and_weights(modelName, summary = False):
	model = model_from_json(open(modelName+'_arch.json', 'r').read())
	if summary == True:
		model.summary()		
	model.load_weights(modelName+'_wgts.h5')
	return model

def modelManager(modelName,x_train, OUTPUT_CLASSES, RETRAIN_MODEL, AI_NAME= 'tinyMedNet', convLayers=None):
	if RETRAIN_MODEL== True:
		if check_model_exists(modelName):
			model = load_model_and_weights(modelName = modelName)
		else:
			nw = networks.get(AI_NAME)
			model = nw(x_train.shape[1:],OUTPUT_CLASSES, convLayers)
			
	else:
		nw = networks.get(AI_NAME)
		model = nw(x_train.shape[1:],OUTPUT_CLASSES, convLayers)
	
	return model
	
def show_model_details(model):
	model.summary()	

def train(	model, x_train, 
			batch_size=1,epochs=1, 
			learning_rate=0.001, callbacks=None,
			class_weights = None, 
			saveBestModel = False, bestModelCond = None, 
			validation_data = None, TRAIN_STEPS = None, TEST_STEPS = None, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], verbose=None, y_train=None, 
		  ):
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
				  loss=loss,
				  metrics=metrics)
	if callbacks is not None:
		if ('tensorboard'in callbacks):
			logdir=os.path.join('log')
			tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
			idx = callbacks.index("tensorboard")	
			callbacks[idx] = tensorboard_callback

	if saveBestModel is not None and saveBestModel==True:
		if bestModelCond is not None and bestModelCond.lower() != 'default':
			print('\n\tENGINE INITIALIZED WITH "SAVING BEST MODEL" and Early Stopping MODE\n')
			earlystop_callback = tf.keras.callbacks.EarlyStopping(
			monitor=bestModelCond['monitor'], 
			min_delta=bestModelCond['min_delta'],
			patience=bestModelCond['patience']
			)
			if callbacks is None:
				callbacks=[earlystop_callback]
			else:
				callbacks.append(earlystop_callback)

	if y_train is not None:
		result = model.fit(x_train, y_train,
				batch_size=batch_size,
				epochs=epochs,
				validation_data=validation_data,
				callbacks=callbacks,
				class_weight = class_weights
				)
	else:
		result = model.fit(x_train,
				epochs=epochs,
				steps_per_epoch=TRAIN_STEPS, validation_steps=TEST_STEPS,
				validation_data=validation_data,
				callbacks=callbacks,
				verbose = verbose,
				class_weight = class_weights
				)		
	return result.history

def plot_training_metrics(result, theme= 'light'):
	result['Epoch'] = np.arange(0, len(result['accuracy']), 1)

	fig = make_subplots(
		rows=2, cols=1,
		shared_xaxes=True,
		x_title = "Epoch Num",
		vertical_spacing=0.1,
		subplot_titles = ('Train and Test Accuracy vs Epochs', 'Train and Test Loss vs Epochs' ),
		specs=[
				[{"type": "scatter"}],
				[{"type": "scatter"}],
			]
	)
	fig.add_trace(go.Scatter(x=result['Epoch'], y=result['accuracy'], mode='lines', name='Train Accuracy'),row=1, col=1)
	fig.add_trace(go.Scatter(x=result['Epoch'], y=result['val_accuracy'], mode='lines', name='Test Accuracy'),row=1, col=1)

	fig.add_trace(go.Scatter(x=result['Epoch'], y=result['loss'], mode='lines', name='Train loss',yaxis="y2"),row=2, col=1)
	fig.add_trace(go.Scatter(x=result['Epoch'], y=result['val_loss'], mode='lines', name='Test loss',),row=2, col=1)
	if theme.lower()=='dark':
		fig.update_layout( template="plotly_dark")
	fig.update_layout(
		yaxis=dict(title="ACCURACY"),
		yaxis2=dict(title="LOSS"),
	)
	fig.show()
	
def predict_labels(model , input, expected_output = None, labelNames=None,top_preds=4):
	"""
	predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
	"""
	output = model.predict(input)
	inds = np.argsort(output)
	expected_output=np.array(expected_output)
	if len(expected_output.shape)>1:
		labelLength = expected_output.shape[1]-1
	else:
		labelLength = expected_output.shape[0]-1
	for j in range(0, len(inds)):
		if expected_output is not None:
			print(20*'=')
			"""
			TODO : Need to fix label lookup when the expected output is not a single number 
			"""
			if len(expected_output.shape)>1:
				print('Expected :',labelNames[expected_output[j][labelLength]])
			else:
				print('Expected :',labelNames[expected_output[j]])
			print(20*'-')
		for i in range(top_preds):
			print(labelNames[inds[j][-1-i]],':',str(round(output[j][inds[j][-1-i]]*100,2)), '%')

def decode_predictions(pred,labelNames, top_preds=4, retType = 'tuple'):
	if retType.lower() == 'tuple':
		numpredDict =[0 for x in range(0,len(pred))]
	else:
		numpredDict ={} 
	for j in range(0, len(pred)):
		predArr =[]
		indexes = np.argsort(pred)
		max_preds = (top_preds, len(labelNames))[len(labelNames) < top_preds] 
		for i in range(max_preds):
			if retType.lower() == 'tuple':
				pDct = (i+1, labelNames[indexes[j][-1-i]], round(pred[j][indexes[j][-1-i]]*100,2))
			elif retType.lower() == 'dict':				
				pDct = {}
				pDct['id']=i+1
				pDct['label'] = labelNames[indexes[j][-1-i]]
				pDct['score'] = round(pred[j][indexes[j][-1-i]]*100,2)
			predArr.append(pDct)
		numpredDict[j] = predArr 
	if retType.lower() == 'tuple':
		return numpredDict
	else:
		resDict ={}
		resDict['Predictions']=numpredDict
		return resDict		

def _get_metrics_evals(TSet,test_predictions,labelNames=None,returnPlot = False, showPlot= False , printStat = False, modelName= 'model', pdfName= None, **kwargs):
	if hasattr(TSet, 'data'):
		labelNames = TSet.labelNames if labelNames is None else labelNames
		generate_evaluation_report(labelNames, test_predictions, groundTruth=TSet.labels,  generator=None , printStat = printStat, returnPlot = returnPlot, showPlot= showPlot ,modelName=modelName, pdfName=pdfName, **kwargs)
	elif hasattr(TSet, 'generator'):
		labelNames = dataprc.safe_labelmap_converter(TSet.labelMap) if labelNames is None else labelNames
		generate_evaluation_report(labelNames, test_predictions, groundTruth=None,  generator=TSet.generator , printStat = printStat, returnPlot = returnPlot, showPlot= showPlot , modelName=modelName,pdfName=pdfName,  **kwargs)	
	

# class _dsDict(dict):
#     pass

class INFERENCE_ENGINE(object):
	"""
		TODO: Need to add Metaloader support
	"""
	def __init__(self,modelName,testSet = None, classNames = None):
		self.result = {}
		self.modelName = modelName
		self.testSet = testSet
		self.preProcessor = None
		if modelName is not None:
			self.model = load_model_and_weights(modelName = modelName)
		self.labelNames = classNames
	
	def load_model_and_weights(self, modelName):
		self.model = load_model_and_weights(modelName = modelName)
	
	def load_network(self, fileName):
		self.model = model_from_json(open(modelName+'_arch.json', 'r').read())

	def load_weights(self, wgtfileName):
		self.model = self.model.load_weights(wgtfileName)
	
	def preprocessor_from_meta(self, metaFile=None):
		if self.modelName is not None:
			self.preProcessor= dataprc.InputProcessorFromMeta(self.modelName)
			self.labelNames = self.preProcessor.labels
	#@timeit
	def predict(self, input):
		if self.preProcessor is not None:
			input = self.preProcessor.processImage(input)
			return self.model.predict(input)
		else:
			if self.labelNames is None:
				if hasattr(input, 'labelNames'):
					self.labelNames = input.labelNames if self.labelNames is None else self.labelNames
			if isinstance(input,np.ndarray):
				return self.model.predict(input)
			elif hasattr(input, 'data'):
				return self.model.predict(input.data)
			elif hasattr(input, 'generator'):
				#self.labelNames =  dataprc.safe_labelmap_converter(input.labelMap) if labelNames is None else labelNames
				return self.model.predict(input.generator)
			else:
				return self.model.predict(input)

	#@timeit
	def predict_pipeline(self, input):
		'''
		Slightly Faster version of predict. Useful for deployment.
		'''
		img = self.preProcessor.processImage(input)
		return self.model.predict(img)

	def decode_predictions(self, pred, top_preds=4, retType = 'tuple'):
		if retType.lower() == 'tuple':
			numpredDict =[0 for x in range(0,len(pred))]
		else:
			numpredDict ={} 
		for j in range(0, len(pred)):
			predArr =[]
			indexes = np.argsort(pred)
			max_preds = (top_preds, len(self.labelNames))[len(self.labelNames) < top_preds] 
			for i in range(max_preds):
				if retType.lower() == 'tuple':
					pDct = (i+1, self.labelNames[indexes[j][-1-i]], round(pred[j][indexes[j][-1-i]]*100,2))
				elif retType.lower() == 'dict':				
					pDct = {}
					pDct['id']=i+1
					pDct['label'] = self.labelNames[indexes[j][-1-i]]
					pDct['score'] = round(pred[j][indexes[j][-1-i]]*100,2)
				predArr.append(pDct)
			numpredDict[j] = predArr 
		if retType.lower() == 'tuple':
			return numpredDict
		else:
			resDict ={}
			resDict['Predictions']=numpredDict
			return resDict	

	def getLayerNames(self):
		return [x.name for x in self.model.layers]

	def summary(self):
		return self.model.summary()

	def generate_evaluation_report(self, testSet = None, predictions = None, printStat = False,returnPlot = False, showPlot= False, pdfName =None, **kwargs):
		if testSet is None:
			testSet = self.testSet
		if predictions is None:
			predictions = self.predict(testSet)
		if pdfName is None:
			pdfName = self.modelName
		_get_metrics_evals(testSet, predictions, printStat = printStat,returnPlot = returnPlot, showPlot= showPlot,modelName=self.modelName, pdfName = pdfName, **kwargs)
		print('Report Generated at Path:\n\t', os.path.abspath(pdfName)+'_report.pdf')

	def explain(self,input, predictions=None, layer_to_explain='CNN3', classNames = None, selectedClasses=None, expectedClass = None, showPlot=False):
		if predictions is None:
			predictions = self.predict(input)#.model.predict(input)
		if classNames is None and self.labelNames is None:
			print('Error: No Labels Provides')
		elif classNames is not None:
			labels = classNames
		elif self.labelNames is not None:
			labels = self.labelNames

		if selectedClasses is None:
			selectedClasses = labels

		predict_with_gradcam(model=self.model, imgNP=input, predictions =predictions, labels=labels, 
							selected_labels = selectedClasses,layer_name=layer_to_explain , expected = expectedClass, showPlot=showPlot)


class TRAIN_ENGINE(INFERENCE_ENGINE):
	"""

	"""
	def __init__(self, modelName=None):
		super().__init__(modelName)

	def train_and_save_model(self,AI_NAME, MODEL_SAVE_NAME, trainSet, testSet, OUTPUT_CLASSES, RETRAIN_MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE, convLayers=None,SAVE_BEST_MODEL=True, BEST_MODEL_COND=None, callbacks=None, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], showModel = False,CLASS_WEIGHTS=None):
		""""
		CLASS_WEIGHTS: Dictionary containing class weights for model.fit()
		"""
		self.testSet = testSet
		self.modelName = MODEL_SAVE_NAME
		self.test_predictions = None
		if hasattr(trainSet, 'data'):
			self.labelNames = trainSet.labelNames
			self.model = modelManager(AI_NAME= AI_NAME, convLayers= convLayers, modelName = MODEL_SAVE_NAME, x_train = trainSet.data, OUTPUT_CLASSES = OUTPUT_CLASSES, RETRAIN_MODEL= RETRAIN_MODEL)
			print(self.model.summary()) if showModel else None
			
			self.result = train(self.model, trainSet.data, y_train= trainSet.labels, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, validation_data=(testSet.data, testSet.labels), callbacks=callbacks, saveBestModel= SAVE_BEST_MODEL, bestModelCond = BEST_MODEL_COND, TRAIN_STEPS = None, TEST_STEPS = None, loss = loss, metrics = metrics,class_weights=CLASS_WEIGHTS)#['tensorboard'])
			#self.model.evaluate(testSet.data, testSet.labels)
			
			dataprc.metaSaver(trainSet.labelMap, trainSet.labelNames,  normalize=trainSet.normalize,
							  rescale =None,
							  network_input_dim =trainSet.network_input_dim, samplingMethodName=trainSet.samplingMethodName, outputName= MODEL_SAVE_NAME)
		else:
			networkDim = np.zeros((1,)+trainSet.generator.image_shape)
			self.labelNames = dataprc.safe_labelmap_converter(trainSet.labelMap)
			self.model = modelManager(AI_NAME= AI_NAME, convLayers= convLayers, modelName = MODEL_SAVE_NAME, x_train = networkDim, OUTPUT_CLASSES = OUTPUT_CLASSES, RETRAIN_MODEL= RETRAIN_MODEL)
			
			print(self.model.summary()) if showModel else None
			self.result = train(self.model, trainSet.generator, batch_size=None, epochs=EPOCHS, learning_rate=LEARNING_RATE, validation_data=testSet.generator, callbacks=callbacks, saveBestModel= SAVE_BEST_MODEL, bestModelCond = BEST_MODEL_COND, TRAIN_STEPS = trainSet.STEP_SIZE, TEST_STEPS = testSet.STEP_SIZE, loss = loss, metrics =metrics, verbose=1,class_weights=CLASS_WEIGHTS)#['tensorboard'])
			#self.model.evaluate(testSet.generator,steps =  testSet.STEP_SIZE)
			dataprc.metaSaver(trainSet.labelMap, self.labelNames, normalize= None,
							 rescale = trainSet.generator.image_data_generator.rescale,
							 network_input_dim =trainSet.generator.image_shape, samplingMethodName=None, outputName= MODEL_SAVE_NAME)

		save_model_and_weights(self.model, outputName= MODEL_SAVE_NAME)

	def plot_train_acc_loss(self):
		plot_training_metrics(self.result)
