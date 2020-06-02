# Example: Image Segmentation (Cell Membrane)
The library currently supports binary segmentation only. The dataset to perform imgage segmentation can be downloaded from [here](https://drive.google.com/file/d/1mz-phQkZxij3WOrXxjGpsKHtX5zbirgm/view)

### Import libraries
```python
import os
import medicalai as ai
import tensorflow as tf
```
### Define the hyperparameters
Specify the dataset folder which further contains test & train folders each with n class object folders
```python
datasetFolderPath = "../data/membrane"
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
```python
augment = ai.AUGMENTATION(rotation_range = 12, 
                          fill_mode='nearest', 
                          width_shift_range=0.1, 
                          height_shift_range=0.1, 
                          brightness_range = (0.9, 1.1), 
                          zoom_range=(0.85, 1.15), 
                          rescale= 1./255)
```
Load your data from folder using datasetGenFromFolder if your data is in folder structured form
```python
dsHandler = ai.segmentaionGenerator(folder=datasetFolderPath,targetDim=(IMG_HEIGHT,IMG_WIDTH), 
                                        augmentation=augment, class_mode=None,
                                        batch_size = 1,color_mode="grayscale",
                                        image_folder_name = "image", mask_folder_name = "label")
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
trainer = ai.TRAIN_ENGINE()
trainer.train_and_save_segmentation(AI_NAME=AI_NAME,
                                    MODEL_SAVE_NAME = MODEL_SAVE_NAME, 
                                    trainSet=trainGen,inputSize = (256,256,1),
                                    BATCH_SIZE= BATCH_SIZE, EPOCHS= EPOCHS, 
                                    LEARNING_RATE= LEARNING_RATE, SAVE_BEST_MODEL = SAVE_BEST_MODEL, 
                                    BEST_MODEL_COND= BEST_MODEL_COND, callbacks = None,
                                    convLayers= convLayers, showModel = False
                             
                           )
```
Use the above model to predict segmentation masks
```python
infEngine = ai.INFERENCE_ENGINE(MODEL_SAVE_NAME)
predsG = infEngine.predict_segmentation(testGen)
```
Save the predicted segmentation masks in a folder
```python
infEngine.saveResult(save_path='results',npyfile=predsG)
```            
![Original Image](http://drive.google.com/uc?export=view&id=1EO6oOZKFmXYnA-Yg26Amqygo-DRd98qI)
![Predicted Mask](http://drive.google.com/uc?export=view&id=14le8KHqNSB38BKX-3qBEwq8dT2aclRzv)         
            
            
            
            
