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
from albumentations import Compose
import albumentations.augmentations.transforms as augmentations

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>1:
	MULTI_GPU_MODE= True
	print('[INFO]: Medicalai activated with MultiGPU Mode')
else:
	MULTI_GPU_MODE= False
GPU_to_Use = 'all'	
	
def create_model_output_folder(outputName):
	"""
	Creates model output folder if model doesn't exist.

	# Arguments
		outputName: (Type - `filepath`): name of the folder where model needs to be created.

	# Returns
		None: None

	"""
	if "\\" in outputName:
		outputName=outputName.replace('\\','/')
	folder= outputName.replace(outputName.split('/')[-1],'')
	
	if "/" in folder:
		if not os.path.exists(folder):
			os.makedirs(folder)	

def check_model_exists(outputName):
	"""
	Checks if the given model's network file exists or not. 
	Model name expected is `modelName + _arch.json`.

	# Arguments
		outputName: (Type - `filepath`): model name to check.

	# Returns
		Bool: If model network exists returns `True` else `False`.

	"""	
	if not os.path.exists(outputName+'_arch.json'):
		print("\n\n\n \t[INFO]: Model Doesnt Exist \n\n\n")
		return False
	else:
		print("\n\n\n \t[INFO]: Using Model: {:}.json \n\n\n".format(outputName))
		return True
		
def save_model_and_weights(model,outputName):
	"""
	Saves the passed model to MedicalAI Format. Accepts a model and converts to MedicalAI Format.
	Produces weight file (`outputName + _wgts.h5`) and network file (`outputName + _arch.json`) 

	!!! danger "IMPORTANT"
		DO NOT PASS ANY EXTENTION TO `outputName` argument

	# Arguments
		model: (Type - `model` class): MedicalAI/Keras/Tensorflow 2.0+ model class.
		outputName: (Type - `filepath`): model path/name to save.

	# Returns
		None: None

	"""	
	create_model_output_folder(outputName)
	model.save_weights(outputName+'_wgts.h5')
	open(outputName+'_arch.json', 'w').write(model.to_json())


def load_model_and_weights(modelName, summary = False):
	"""
	Loads model from the given filepath. 
	Function Expects weight file (`modelName + _wgts.h5`) and network file (`modelName + _arch.json`).

	!!! danger "NOTE"\n\t\tDO NOT PASS ANY EXTENTION TO `outputName` argument

	For Example:
	```Python
	# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
	# Then `modelName=devModel/testmodel1`

	modelName = 'devModel/testmodel1'
	load_model_and_weights(modelName, summary = False)

	load_model_and_weights(modelName='devModel/testmodel1')

	load_model_and_weights('devModel/testmodel1', summary = True)
	```

	
	# Arguments
		modelName: (Type - `filepath`): model path/name to save.
		summary: (Type - `Bool`): Show loaded network architecture and parameter summary.
		
	# Returns
		model: (Type - `model` class): MedicalAI/Keras/Tensorflow 2.0+ model class.

	"""	
	model = model_from_json(open(modelName+'_arch.json', 'r').read())
	if summary == True:
		model.summary()		
	model.load_weights(modelName+'_wgts.h5')
	return model

def modelManager(modelName,x_train, OUTPUT_CLASSES, RETRAIN_MODEL, AI_NAME= 'tinyMedNet', **kwargs):
	"""
	Model manager is used to build new model for given networks/AI or reload existing AI model. 
	This function can be used to retrain existing models or create new models.


	!!! danger "IMPORTANT"
		DO NOT PASS ANY EXTENTION TO `modelName` argument

	# Arguments
		modelName: (Type - `filepath`): model path/name to load existing model or create new model.
		x_train: (Type - `numpy.array`): training dataset - expected shape [num_samples*dimension_of_input].
		OUTPUT_CLASSES: (Type - `Int`): Number of unique classes in dataset.
		RETRAIN_MODEL: (Type - `Bool`): Whether to retrain existing model. If set to `True` and model does not
										exist, then it creates a new model and subsequent runs will retrain model.
		AI_NAME: (Type - `String` or `Custom Network Class`): Select AI Networks from existing catalogue in MedicalAI.
										See AI_NAME Page for More Details.	
		convLayers: (Type - `Int`): [Optional] Default is None. Only applicable for certain networks where convolution 
										layers are reconfigurable. This parameter can be used to change the num of conv 
										layers in Network. See AI_NAME Page for More Details.	

	# Returns
		model: (Type - `model` class): MedicalAI/Keras/Tensorflow 2.0+ model class.

	See Also:
		TRAIN_ENGINE, INFERENCE_ENGINE
	"""	
	if RETRAIN_MODEL== True:
		if check_model_exists(modelName):
			model = load_model_and_weights(modelName = modelName)
		else:
			create_model_output_folder(modelName)
			nw = networks.get(AI_NAME)
			model = nw(x_train.shape[1:],OUTPUT_CLASSES, **kwargs)
			
	else:
		nw = networks.get(AI_NAME)
		model = nw(x_train.shape[1:],OUTPUT_CLASSES, **kwargs)
	
	return model
	
def show_model_details(model):
	"""
	Show model network structure and print parameters summary.

	# Arguments
		model: (Type - `model` class): MedicalAI/Keras/Tensorflow 2.0+ model class.

	# Returns
		None: None; Prints the model summary
	"""	
	model.summary()	

def train(	model, x_train, 
			batch_size=1,epochs=1, 
			callbacks=None,
			class_weights = None, 
			saveBestModel = False, bestModelCond = None, 
			validation_data = None, TRAIN_STEPS = None, TEST_STEPS = None, 
			verbose=None, y_train=None, workers = 1 
		  ):
	if callbacks is not None:
		if ('tensorboard'in callbacks):
			logdir=os.path.join('log')
			tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
			idx = callbacks.index("tensorboard")	
			callbacks[idx] = tensorboard_callback

	if saveBestModel is not None and saveBestModel==True:
		if bestModelCond is not None and bestModelCond.lower() != 'default':
			print('\n[INFO]: ENGINE INITIALIZED WITH "SAVING BEST MODEL" and Early Stopping MODE\n')
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
				class_weight = class_weights,
				workers =workers
				)
	else:
		result = model.fit(x_train,
				epochs=epochs,
				steps_per_epoch=TRAIN_STEPS, validation_steps=TEST_STEPS,
				validation_data=validation_data,
				callbacks=callbacks,
				verbose = verbose,
				class_weight = class_weights,
				workers = workers
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
	```Python
	predict_labels(model , input, expected_output = expected_output, labelNames=classNames,top_preds=4)
	```
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
		generate_evaluation_report(labelNames, test_predictions, groundTruth=None,	generator=TSet.generator , printStat = printStat, returnPlot = returnPlot, showPlot= showPlot , modelName=modelName,pdfName=pdfName,  **kwargs)	
	
class INFERENCE_ENGINE(object):
	"""
		Initializes Inference Engine to perform inference/prediction on a trained model. 
		Can be used during production. 

	# Arguments
		modelName: (Type - `filepath`): model path/name to load existing model or create new model.
		testSet: (Type - `numpy.array` or `generator`): [Optional] : Test/Validation Dataset either as generator
											or numpy array. Only passed if performing evaluation. No need to set 
											this during production.
		classNames: (Type - `list` or `numpy.array`): [Optional] : classNames or labelNames for the dataset. 
		
	# Returns
		INFERENCE_ENGINE Object: If `modelName` is supplied, returns an object with loaded model.

	"""
	def __init__(self,modelName=None,testSet = None, classNames = None):
		self.result = {}
		self.modelName = modelName
		self.testSet = testSet
		self.preProcessor = None
		if modelName is not None:
			self.model = load_model_and_weights(modelName = modelName)
		self.labelNames = classNames
		print('[INFO]: Initialized -',self.__class__.__name__)
		if self.__class__.__name__ == 'INFERENCE_ENGINE':
			try:
				self.preprocessor_from_meta()
			except:
				print('[INFO]: Meta File Not Found - Not Initializing Preprocessor from Meta')
	
	def load_model_and_weights(self, modelName, summary=False):
		"""
		Loads model from the given filepath. 
		Function Expects path to weight file (`modelName + _wgts.h5`) and network file (`modelName + _arch.json`).

		!!! info "NOTE"
			You can use `load_network` and `load_weights` if the model files are in MedicalAI Format.

		!!! danger "WARNING"
			DO NOT PASS ANY EXTENTION TO `outputName` argument

		For Example:
		```Python
		# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
		# Then `modelName=devModel/testmodel1`

		modelName = 'devModel/testmodel1'
		infEngine = INFERENCE_ENGINE()
		infEngine.load_model_and_weights(modelName)

		infEngine.load_model_and_weights(modelName, summary = True)
		```

		
		# Arguments
			modelName: (Type - `filepath`): model path/name to load.
			summary: (Type - `Bool`): [Optional] : `Default = False`. Show loaded network architecture 
										and parameter summary.
			
		# Returns
			None: Intializes Object with model.
			
		"""
		self.model = load_model_and_weights(modelName = modelName, summary=summary)
	
	def load_network(self, fileName):
		"""
		Loads network from given filepath. Function Expects path to network file with `.json` extension.

		!!! info "NOTE"
			Use this function only if the model files are not in MedicalAI Format. 

		Example:
		```Python

		networkFile = 'devModel/testmodel1.json'

		infEngine = INFERENCE_ENGINE()
		infEngine.load_network(networkFile)
		```

		
		# Arguments
			modelName: (Type - `filepath`): model network path/name to load. File should have `.json` extension.
			
		# Returns
			None: Intializes Object with model network initialized. After this model weights can be loaded.
			
		"""
		if '.json' not in modelName:
			self.model = model_from_json(open(modelName+'_arch.json', 'r').read())
		else:
			self.model = model_from_json(open(modelName, 'r').read())

	def load_weights(self, wgtfileName):
		"""
		Loads weight from given filepath. Function Expects path to weight file with `.h5` extension.

		!!! danger "NOTE"
			Use this function only if the model files are not in MedicalAI Format. 
			Before calling this function, network needs to loaded using `load_network` function.

		Example:
		```Python

		networkFile = 'devModel/testmodel1.json'
		wgtFile = 'devModel/testmodel1.h5'

		infEngine = INFERENCE_ENGINE()
		infEngine.load_network(networkFile)
		infEngine.load_weights(wgtFile)
		```

		
		# Arguments
			wgtfileName: (Type - `filepath`): model weight filepath/name to load. File should have `.h5` extension.
			
		# Returns
			None: Intializes Object with model loaded with weights.
		"""
		self.model = self.model.load_weights(wgtfileName)


	def preprocessor_from_meta(self, metaFile=None):
		"""
		Loads preprocessor parameter and initializes preprocessor from meta file generated by MedicalAI. 
		
		If the model is trained using this framework, then the metafile is automatically available and initialized.

		!!! danger "WARNING"
			If model is not trained using this framework, then one can use this engine by creating metafile
			similar to one generated by this framework. Please see repo for more details.

		Example:
		```Python
		# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
		# Then `modelName=devModel/testmodel1`

		modelName = 'devModel/testmodel1'
		infEngine = INFERENCE_ENGINE()
		infEngine.load_model_and_weights(modelName)

		# There is no need to perform this op if model trained using this framework. It is automatically Initialized.
		# There is no need to pass modelName if the model is trained using framework
		infEngine.preprocessor_from_meta() 
		
		infEngine.preprocessor_from_meta(metaFile='myMetaFile.json') #Else pass the metafile
		```

		
		# Arguments
			metaFile: (Type - `filepath`): [Optional] : if no parameter is passed, then it will look for 
											`modelname + _meta.json` file. If modelname is set during 
											INFERENCE_ENGINE initialization, then it automatically handles this.
			
		# Returns
			None: Intializes Object with Preprocessor into process pipeline.
		"""
		if self.modelName is not None:
			self.preProcessor= dataprc.InputProcessorFromMeta(self.modelName)
			self.labelNames = self.preProcessor.labels
		else:
			self.preProcessor= dataprc.InputProcessorFromMeta(metaFile)
			self.labelNames = self.preProcessor.labels

	#@timeit
	def predict(self, input, verbose=1, safe=False  , workers= 1):
		"""
		Peform prediction on Input. Input can be Numpy Array or Image or Data Generator (in case of Test/Validation). 

		Example:
		```Python
		# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
		# Then `modelName=devModel/testmodel1`

		modelName = 'devModel/testmodel1'

		infEngine = INFERENCE_ENGINE()
		infEngine.load_model_and_weights(modelName)
		infEngine.preprocessor_from_meta()

		# Predict an input image
		infEngine.predict(input = 'test.jpg')
		```

		
		# Arguments
			input: (Type - `numpy.array`|`imagePath`|`generator` ): Can be single image file or numpy array of multiple
											images or data generator class.
			
		# Returns
			Numpy.Array: of Predictions. Shape of Output [Number of Inputs, Number of Output Classes in Model]
		"""
		if hasattr(self, 'workers'):
			workers = self.workers  
		else:
			workers = workers
		if safe:
			if hasattr(input, 'generator') and hasattr(input, 'STEP_SIZE'):
				return self.model.predict(input.generator, steps=input.STEP_SIZE, verbose=1, workers=workers)
			elif hasattr(input, 'image_data_generator'):
				return self.model.predict(input,  steps =(input.n/input.batch_size), verbose=1, workers=workers)
		else:			
			if hasattr(input, 'generator') and hasattr(input, 'STEP_SIZE'):
				return self.model.predict(input.generator, verbose=1, workers=workers)
			elif hasattr(input, 'image_data_generator'):
				return self.model.predict(input,  verbose=1, workers=workers)
			elif hasattr(input, 'data') and not isinstance(input,np.ndarray):
				return self.model.predict(input.data, verbose=verbose, workers=workers)
			else:
				if self.preProcessor is not None:
					input = self.preProcessor.processImage(input)
					return self.model.predict(input, verbose=verbose, workers=workers)
				else:
					if self.labelNames is None:
						if hasattr(input, 'labelNames'):
							self.labelNames = input.labelNames if self.labelNames is None else self.labelNames
					if isinstance(input,np.ndarray):
						return self.model.predict(input, verbose=verbose, workers=workers)
					else:
						return self.model.predict(input, verbose=verbose, workers=workers)

	#@timeit
	def predict_pipeline(self, input):
		"""
		Slightly Faster version of predict. Useful for deployment. Do not use `INFERENCE_ENGINE.predict` in production.
		Peform prediction on Input. Input can be Numpy Array or Image or Data Generator (in case of Test/Validation). 

		Example:
		```Python
		# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
		# Then `modelName=devModel/testmodel1`

		modelName = 'devModel/testmodel1'

		infEngine = INFERENCE_ENGINE()
		infEngine.load_model_and_weights(modelName)
		infEngine.preprocessor_from_meta()

		# Predict an input image
		infEngine.predict_pipeline(input = 'test.jpg')
		```

		
		# Arguments
			input: (Type - `numpy.array`|`imagePath`|`generator` ): Can be single image file or numpy array of multiple
											images or data generator class.
			
		# Returns
			Numpy.Array: of Predictions. Shape of Output [Number of Inputs, Number of Output Classes in Model]
		"""
		img = self.preProcessor.processImage(input)
		return self.model.predict(img)

	def predict_with_labels(self, input,top_preds=4, retType = 'tuple'):
		"""
		Returns Decodes predictions with label/class names with output probabilites
		"""
		pred = self.predict(input)
		return self.decode_predictions( pred, top_preds= top_preds, retType = retType)

	def decode_predictions(self, pred, top_preds=4, retType = 'tuple'):
		"""
		Returns Decodes predictions with label/class names with output probabilites. 
		During production this can be used to return a json serializable dictionary instead of tuple.

		Example:
		```Python
		# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
		# Then `modelName=devModel/testmodel1`

		modelName = 'devModel/testmodel1'

		infEngine = INFERENCE_ENGINE()
		infEngine.load_model_and_weights(modelName)
		infEngine.preprocessor_from_meta()

		# Predict an input image
		pred = infEngine.predict_pipeline(input = 'test.jpg')
		pred_tuple = infEngine.decode_predictions(pred, top_preds=2)

		# Get a json serializable dictionary instead of tuple
		pred_dict  = infEngine.decode_predictions(pred, top_preds=2, retType = 'dict')
		
		```

		# Arguments
			pred: (Type - `numpy.array`): Prediction output of either `INFERENCE_ENGINE.predict` or 
										 `INFERENCE_ENGINE.predict_pipleline`. 
			top_preds: (Type - `Integer`): [Optional] : `Default = 4` - Number of top prediction to return. If the number is 
										  set to higher than number of classes in network, it returns all predictions.
			retType: (Type - `String`): [Optional] : `Default = tuple`. Options - [`dict` or `tuple`]. `Dict` helpful in production.

		# Returns
			Tuple or Dict: of Predictions with probabilities. Shape of Output [Number of Inputs, Max(top_preds,Number of Output Classes in Model)]
		"""
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
		"""
		Get the layer names of the network. Useful for when using Explainable-AI function as it expects `layer name` as argument.

		Example:
		```Python
		# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
		# Then `modelName=devModel/testmodel1`

		modelName = 'devModel/testmodel1'

		infEngine = INFERENCE_ENGINE()
		infEngine.load_model_and_weights(modelName)

		# Print the Layer Names
		print('\n'.join(infEngine.getLayerNames()))
		```

		"""
		return [x.name for x in self.model.layers]

	def summary(self):
		"""
		Show model network structure and print parameters summary.

		# Arguments
			None: None

		# Returns
			None: None; Prints the model summary
		"""	
		return self.model.summary()

	def generate_evaluation_report(self, testSet = None, predictions = None, printStat = True,returnPlot = False, showPlot= False, pdfName =None, **kwargs):
		"""
		Generate a comprehensive PDF report with model sensitivity, specificity, accuracy, confidence intervals,
		ROC Curve Plot, Precision Recall Curve Plot, and Confusion Matrix Plot for each class.
		This function can be used when evaluating a model with Test or Validation Data Set.

		Example:

		```Python
		# Load Dataset
		trainSet,testSet,labelNames =ai.datasetFromFolder(datasetFolderPath, targetDim = (224,224)).load_dataset()

		# Intialize Inference Engine
		infEngine = ai.INFERENCE_ENGINE(MODEL_SAVE_NAME)

		# Preform Prediction on DataSet
		predsG = infEngine.predict(testSet.data)

		# Generate Report
		infEngine.generate_evaluation_report(testSet,predictions = predsG , pdfName = "expt_evaluation_report.pdf")

		```

		Alternatively:
		```Python
		# Load Dataset
		trainSet,testSet,labelNames =ai.datasetFromFolder(datasetFolderPath, targetDim = (224,224)).load_dataset()

		# Intialize Inference Engine
		infEngine = ai.INFERENCE_ENGINE(MODEL_SAVE_NAME)

		# Generate Report - If predictions are not passed, then automatically prediction is performed.
		infEngine.generate_evaluation_report(testSet, pdfName = "expt_evaluation_report.pdf")

		```
		# Arguments
			testSet: (Type - `numpy.array` or `generator`) : Test Data Set to perform evaluation on.
			predictions: (Type - `numpy.array`): [Optional] : Prediction output of either `INFERENCE_ENGINE.predict` or 
										 `INFERENCE_ENGINE.predict_pipleline`. If this parameter is not set, then prediction 
										 is perfomred internally and evaluation report is generated.
			pdfName: (Type - `Bool`): [Optional] : `Default = ModelName + _report.pdf` - Pdf Output Name.
			printStat: (Type - `Bool`): [Optional] : `Default = False` - Print Statistics on console.
			returnPlot: (Type - `Bool`): [Optional] : `Default = False` - Return Plot Figure Handle.
			showPlot: (Type - `Bool`): [Optional] : `Default = False` - Show Plot figure.

		# Returns
			None or Plot Handle: If `returnPlot = True` then Plot Handle will be returned else None.
		"""
		if testSet is None:
			testSet = self.testSet
		if predictions is None:
			predictions = self.predict(testSet)
		if pdfName is None:
			pdfName = self.modelName
		_get_metrics_evals(testSet, predictions, printStat = printStat,returnPlot = returnPlot, showPlot= showPlot,modelName=self.modelName, pdfName = pdfName, **kwargs)
		print('[INFO]: Report Generated at Path:\n\t', os.path.abspath(pdfName)+'_report.pdf')

	def explain(self,input, predictions=None, layer_to_explain='CNN3', classNames = None, selectedClasses=None, expectedClass = None, showPlot=False):
		"""
		Explains a model layer with respect to Input and Output using Grad-cam. Basically, see what the AI is seeing to arrive at 
		a certain prediction. More methods to be updated in next versions.

		``` Python
		# Load a sample
		image = load(Image)

		# Intialize Inference Engine
		infEngine = ai.INFERENCE_ENGINE(MODEL_SAVE_NAME)

		# Print Layer Names
		print('\n'.join(infEngine.getLayerNames()))

		# If predictions are not passed, then automatically prediction is performed. You can perform prediction first then pass
		  to the below function. Pass one of the layer name from above output to `layer_to_explain`.
		infEngine.explain(image, layer_to_explain='CNN3')
		```

		# Arguments
			input: (Type - `numpy.array` or `image`) : Input to perform explanation on. For safety, pass single or few samples only.
			predictions: (Type - `numpy.array`): [Optional] : Prediction output of either `INFERENCE_ENGINE.predict` or 
										 `INFERENCE_ENGINE.predict_pipleline`. If this parameter is not set, then prediction 
										 is perfomred internally and explanation is generated.
			layer_to_explain: (Type - `String`):  Layer to explain.
			classNames: (Type - `Numpy.Array` or `List`): [Optional] : `Default = None| Loaded from Meta File` - Class Names or Label Names of Dataset.
			selectedClasses: (Type - `Bool`): [Optional] : `Default = None` - Explain only few subset of Class Names. If `None` then all classes will be explained.
			expectedClass: (Type - `Bool`): [Optional] : `Default = None` - Expected Label/Class Name for the Input.

		# Returns
			None: Shows a plot figure with explanations.

		"""
		if predictions is None:
			try:
				img = self.preProcessor.processImage(input)
				predictions = self.predict(img)#.model.predict(input)
			except:
				predictions = self.predict(input)
				
		if classNames is None and self.labelNames is None:
			print('[ERROR]: No Labels Provides')
		elif classNames is not None:
			labels = classNames
		elif self.labelNames is not None:
			labels = self.labelNames

		if selectedClasses is None:
			selectedClasses = labels

		if layer_to_explain is None:
			myLayers = self.getLayerNames()
			layer_to_explain = myLayers[0]
		predict_with_gradcam(model=self.model, imgNP=input, predictions =predictions, labels=labels, 
							selected_labels = selectedClasses,layer_name=layer_to_explain , expected = expectedClass, showPlot=showPlot)


class TRAIN_ENGINE(INFERENCE_ENGINE):
	"""
		Initializes Training Engine to perform training/prediction. TRAIN_ENGINE is a superclass of INFERENCE_ENGINE.
		Meaning, all the methods and functions of INFERENCE_ENGINE are available with TRAIN_ENGINE with additional methods of 
		its own.

	# Arguments
		modelName: (Type - `filepath`): [Optional] model path/name to load existing model or create new model.
		
	# Returns
		TRAIN_ENGINE Object: Ready to Train a given dataset.

	"""
	def __init__(self, modelName=None):
		super().__init__(modelName)
		
	def train_and_save_model(self,AI_NAME, MODEL_SAVE_NAME, trainSet, testSet, OUTPUT_CLASSES, RETRAIN_MODEL,  EPOCHS, 
							BATCH_SIZE=32, LEARNING_RATE=0.0001, convLayers=None,SAVE_BEST_MODEL=False, BEST_MODEL_COND=None, 
							callbacks=None, loss = 'sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001),
							metrics = ['accuracy'], showModel = False, workers = 1,
							CLASS_WEIGHTS=None, **kwargs,):
		""""
		Main function that trains and saves a model. This automatically builds new model for given networks/AI or reload existing AI model. 
		This function can be used to retrain existing models or create new models.

		!!! danger "IMPORTANT"
			DO NOT PASS ANY EXTENTION TO `MODEL_SAVE_NAME` argument
		
		USAGE:

		```Python

		# Set Parameters
		AI_NAME = 'MobileNet_X'
		MODEL_SAVE_NAME = 'testModel1'
		
		OUTPUT_CLASSES = 10
		RETRAIN_MODEL = True
		EPOCHS = 10
		BATCH_SIZE = 32
		LEARNING_RATE = 0.0001
		SAVE_BEST_MODEL = False
		BEST_MODEL_COND = None	
		callbacks = None

		# Initialize Train Engine
		trainer = ai.TRAIN_ENGINE()

		# Train and Save Model
		trainer.train_and_save_model(AI_NAME=AI_NAME,  														# AI/Network to Use
									MODEL_SAVE_NAME = MODEL_SAVE_NAME, 										# Target MODEL To Save/Load/Retrain
									trainSet=trainGen, testSet=testGen, OUTPUT_CLASSES=OUTPUT_CLASSES, 		# From Dataset Loader
									RETRAIN_MODEL= RETRAIN_MODEL, BATCH_SIZE= BATCH_SIZE, EPOCHS= EPOCHS, 	# Training Settings
									SAVE_BEST_MODEL = SAVE_BEST_MODEL, 	BEST_MODEL_COND= BEST_MODEL_COND, 	# Early Stopping Settings
									loss='categorical_crossentropy',										# Loss Function
									showModel = False,														# Show Network Summary
									callbacks = callbacks,													# Additional/Advanced Hooks
									)
		```

		# Arguments
			AI_NAME: (Type - `string` or `NetworkInit() class`): Select Network from catalogue (string) or create your own network and pass the class.
			MODEL_SAVE_NAME: (Type - `filepath`): [Optional] model path/name to load existing model or create new model.
			trainSet: (Type - `numpy.array` or `generator`): [Optional] : Training Dataset either as generator or numpy array from `DataLoader` class.
			testSet: (Type - `numpy.array` or `generator`): [Optional] : Test/Validation Dataset either as generator or numpy array 
								from `DataLoader` class.
			OUTPUT_CLASSES: (Type - `Int`): Number of unique classes in dataset.
			RETRAIN_MODEL: (Type - `Bool`): Whether to retrain existing model. If set to True and model does not exist, 
								then it creates a new model and subsequent runs will retrain model.
			BATCH_SIZE: (Type - `Int`): Batch size for Training. If Training fails when using large datasets, try reducing this number. 
			EPOCHS: (Type - `Int`): Number of Epochs to train.
			LEARNING_RATE: (Type - `Float`): [Optional] : Set Learning rate. If not set, optimizer default will be used.
			convLayers: (Type - `Int`): [Optional] Default is None. Only applicable for certain networks where convolution 
											layers are reconfigurable. This parameter can be used to change the num of conv 
											layers in Network. See AI_NAME Page for More Details.	
			SAVE_BEST_MODEL: (Type - `Bool`): [Optional] : `Default: False` - Initializes Training Engine with saving best model feature. 
			BEST_MODEL_COND: (Type - `String` or `Dict`): [Optional] : `Default: None` - Initializes Training Engine with early stopping feature. 
							[Options] -> `Default` or `Dict`. 
							Dict Values Expected:
							'monitor': (Type - `String`): Which Parameter to Monitor. [Options] -> ('val_accuracy', 'val_loss', 'accuracy'), 
							'min_delta': (Type - `Float`): minimum change in the monitored quantity to qualify as an improvement, 
									i.e. an absolute change of less than min_delta, will count as no improvement.
							'patience': (Type - `Int`): number of epochs with no improvement after which training will be stopped.
			loss: (Type - `String`) : `Default: sparse_categorical_crossentropy`, Loss function to apply. Depends on dataprocessor.
										If dataloaders has one-hot encoded labels then use `sparse_categorical_crossentropy` else if 
										labers are encoded then -> `categorical_crossentropy`.
			metrics: (Type - `List`): [Optional] : `Default: ['accuracy']`. Metrics to Monitor during Training.
			showModel: (Type - `Bool`): [Optional] : Whether to show the network summary before start of training.
			CLASS_WEIGHTS: (Type - `Dict`) [Optional] : Dictionary containing class weights for model.fit()
			callbacks: (Type - `Tensorflow Callbacks`): Tensorflow Callbacks can be attacked.

		# Returns
			None: On successful completion saves the trained model.

		"""
		self.workers = workers
		self.testSet = testSet
		self.modelName = MODEL_SAVE_NAME
		self.test_predictions = None
		global MULTI_GPU_MODE, GPU_to_Use
		if hasattr(trainSet, 'data'):
			self.labelNames = trainSet.labelNames
			if MULTI_GPU_MODE and GPU_to_Use.lower()=='all':
				mirrored_strategy = tf.distribute.MirroredStrategy()
				with mirrored_strategy.scope():
					self.model = modelManager(AI_NAME= AI_NAME, convLayers= convLayers, modelName = MODEL_SAVE_NAME, x_train = trainSet.data, OUTPUT_CLASSES = OUTPUT_CLASSES, RETRAIN_MODEL= RETRAIN_MODEL)
					self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
				#BATCH_SIZE *= mirrored_strategy.num_replicas_in_sync
			else:
					self.model = modelManager(AI_NAME= AI_NAME, convLayers= convLayers, modelName = MODEL_SAVE_NAME, x_train = trainSet.data, OUTPUT_CLASSES = OUTPUT_CLASSES, RETRAIN_MODEL= RETRAIN_MODEL)
					self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)				
			print(self.model.summary()) if showModel else None
			print('[INFO]: BATCH_SIZE -',BATCH_SIZE)
			self.result = train(self.model, trainSet.data, y_train= trainSet.labels, batch_size=BATCH_SIZE, epochs=EPOCHS,
									validation_data=(testSet.data, testSet.labels), callbacks=callbacks, saveBestModel= SAVE_BEST_MODEL, 
									bestModelCond = BEST_MODEL_COND, TRAIN_STEPS = None, TEST_STEPS = None, workers = self.workers,
									class_weights=CLASS_WEIGHTS)#['tensorboard'])
			#self.model.evaluate(testSet.data, testSet.labels)
			
			dataprc.metaSaver(trainSet.labelMap, trainSet.labelNames,  normalize=trainSet.normalize,
							  rescale =None,
							  network_input_dim =trainSet.network_input_dim, samplingMethodName=trainSet.samplingMethodName, outputName= MODEL_SAVE_NAME)
		else:
			from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
			if isinstance(trainSet.generator, PrefetchDataset):
				for f,l in trainSet.generator.take(1):
					inpSize = f.numpy().shape
				networkDim = np.zeros((1,)+inpSize[1:])
				networkInputSize = inpSize[1:]
				rescaleValue = 1./255
			else:
				networkDim = np.zeros((1,)+trainSet.generator.image_shape)
				networkInputSize = trainSet.generator.image_shape
				try:
					rescaleValue = trainSet.generator.image_data_generator.rescale
				except:
					rescaleValue = 1./255

			self.labelNames = dataprc.safe_labelmap_converter(trainSet.labelMap)
			if MULTI_GPU_MODE and GPU_to_Use.lower()=='all':
				mirrored_strategy = tf.distribute.MirroredStrategy()
				with mirrored_strategy.scope():
					self.model = modelManager(AI_NAME= AI_NAME, modelName = MODEL_SAVE_NAME, x_train = networkDim, OUTPUT_CLASSES = OUTPUT_CLASSES, RETRAIN_MODEL= RETRAIN_MODEL, **kwargs)
					self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
			else:
					self.model = modelManager(AI_NAME= AI_NAME, modelName = MODEL_SAVE_NAME, x_train = networkDim, OUTPUT_CLASSES = OUTPUT_CLASSES, RETRAIN_MODEL= RETRAIN_MODEL, **kwargs)
					self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)				
			print(self.model.summary()) if showModel else None
			print('[INFO]: BATCH_SIZE -',BATCH_SIZE)
			self.result = train(self.model, trainSet.generator, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=testSet.generator, 
					callbacks=callbacks, saveBestModel= SAVE_BEST_MODEL, bestModelCond = BEST_MODEL_COND, TRAIN_STEPS = trainSet.STEP_SIZE, 
					TEST_STEPS = testSet.STEP_SIZE, verbose=1,class_weights=CLASS_WEIGHTS, workers = self.workers
					)
			#self.model.evaluate(testSet.generator,steps =	testSet.STEP_SIZE)
			dataprc.metaSaver(trainSet.labelMap, self.labelNames, normalize= None,
							 rescale = rescaleValue,
							 network_input_dim =networkInputSize, samplingMethodName=None, outputName= MODEL_SAVE_NAME)

		save_model_and_weights(self.model, outputName= MODEL_SAVE_NAME)

	def plot_train_acc_loss(self):
		"""
		Plot training accuracy and loss graph vs epoch. Generates an interactive graph for inspection.

		USAGE:

		```Python

		# Set Parameters
		AI_NAME = 'MobileNet_X'
		MODEL_SAVE_NAME = 'testModel1'
		
		OUTPUT_CLASSES = 10
		RETRAIN_MODEL = True
		EPOCHS = 10
		BATCH_SIZE = 32
		LEARNING_RATE = 0.0001
		SAVE_BEST_MODEL = False
		BEST_MODEL_COND = None	
		callbacks = None

		# Initialize Train Engine
		trainer = ai.TRAIN_ENGINE()

		# Train and Save Model
		trainer.train_and_save_model(AI_NAME=AI_NAME,  														# AI/Network to Use
									MODEL_SAVE_NAME = MODEL_SAVE_NAME, 										# Target MODEL To Save/Load/Retrain
									trainSet=trainGen, testSet=testGen, OUTPUT_CLASSES=OUTPUT_CLASSES, 		# From Dataset Loader
									RETRAIN_MODEL= RETRAIN_MODEL, BATCH_SIZE= BATCH_SIZE, EPOCHS= EPOCHS, 	# Training Settings
									SAVE_BEST_MODEL = SAVE_BEST_MODEL, 	BEST_MODEL_COND= BEST_MODEL_COND, 	# Early Stopping Settings
									loss='categorical_crossentropy',										# Loss Function
									showModel = False,														# Show Network Summary
									callbacks = callbacks,													# Additional/Advanced Hooks
									)

		trainer.plot_training_metrics()
		```
		# Arguments
			None: None

		# Returns
			None: Opens accuracy vs loss vs epoch plot.
		"""	

		plot_training_metrics(self.result)
