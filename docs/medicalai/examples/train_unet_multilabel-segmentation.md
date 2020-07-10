# Example: Image Segmentation (Camvid Dataset)
The dataset to perform imgage segmentation can be downloaded from [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

### Import libraries
```python
import os
import numpy as np
import medicalai as ai
import tensorflow as tf
```
### Define the hyperparameters
Specify the dataset folder which further contains test & train folders each with n class object folders
```python
datasetFolderPath = "../data/camvid/"
```
Specify the dimensions of image to be fed to network
```python
(IMG_HEIGHT,IMG_WIDTH) = (256,256)
```
Specify the name of the model to be trained on
```python
EXPT_NAME = '1'
AI_NAME = 'unet' 
MODEL_SAVE_NAME = '../model/'+AI_NAME+'/Medical_RSNA_'+str(IMG_HEIGHT)+'x'+str(IMG_WIDTH)+'_'+AI_NAME+'_EXPT_'+str(EXPT_NAME)
```
Specify remaining hyperparamters
```
batch_size = 32
epochs = 10
learning_rate = 0.0001

```
### Define the augmentation for the generator
(The following augmentation is for image only)
```python
augment = ai.AUGMENTATION(rescale= 1./255)
```
Load your data from folder using datasetGenFromFolder if your data is in folder structured form. Make sure ```flag_multi_class``` is set to ```True```
```python
dsHandler = ai.segmentaionGenerator(folder=datasetFolderPath,targetDim=(IMG_HEIGHT,IMG_WIDTH), 
                                        augmentation=augment, class_mode=None,
                                        batch_size = batch_size,
                                        image_folder_name = "image", mask_folder_name = "masks",
                                        flag_multi_class=True)
```
```python
trainGen = dsHandler.load_train_generator()
testGen = dsHandler.load_test_generator()
```
### Train model
Now our image generator is ready to be trained on our model. But first we need to define a tensorflow callback for the model
```python
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                        MODEL_SAVE_NAME+'best.h5', 
                                        verbose=0,
                                        mode='auto', 
                                        save_freq=5,
                                        save_best_only=True,
                                        )
callbacks = [model_checkpoint]
```
```python
TRAIN_STEPS = int(np.ceil(dsHandler.imageGen.generator.n/dsHandler.batch_size))
trainer = ai.TRAIN_ENGINE()
trainer.train_and_save_segmentation(AI_NAME=AI_NAME,
                                    MODEL_SAVE_NAME = MODEL_SAVE_NAME, 
                                    trainSet=trainGen,inputSize = (IMG_HEIGHT,IMG_WIDTH,3),
                                    TRAIN_STEPS=TRAIN_STEPS,
                                    BATCH_SIZE= BATCH_SIZE, EPOCHS= EPOCHS, 
                                    LEARNING_RATE= LEARNING_RATE, SAVE_BEST_MODEL = SAVE_BEST_MODEL, 
                                    callbacks = callbacks,
                                    showModel = False)
                                    
```
Use the above model to predict segmentation masks
```python
infEngine = ai.INFERENCE_ENGINE(MODEL_SAVE_NAME)
predsG = infEngine.predict_segmentation(testGen)
```
Save the predicted segmentation masks in a folder
```python
infEngine.saveResult(save_path='results',npyfile=predsG,num_class=32,flag_multi_class=True)
```            
