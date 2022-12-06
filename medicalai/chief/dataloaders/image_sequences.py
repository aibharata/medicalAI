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
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
import os
import albumentations.augmentations.transforms as AUG
from albumentations import Compose
from .data_utils import *
from ..dataset_analysis import compute_class_freqs
from .dataset_visualize import *


class ImageDatasetSeqFromDF(object):
    def __init__(self, trainDF=None, testDF=None, valDF=None, dataFolder='',
                 inputCol="files", labelCols=['labels'], batch_size=16, 
                 targetDim=(96,96),color_mode="rgb",shuffle=True, seed=21, 
                 class_mode="raw", train_augmentations=Compose([AUG.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
                 test_val_augmentations=Compose([AUG.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
                 class_weights=None, output_range=None,
                 ):
        self.trainDF, self.testDF, self.valDF = trainDF, testDF, valDF
        self.dataFolder = dataFolder
        self.inputCol, self.labelCols = inputCol, labelCols
        self.batch_size = batch_size
        self.targetDim, self.color_mode = targetDim, color_mode
        self.shuffle, self.seed = shuffle, seed
        self.class_mode = class_mode
        self.train_augment = train_augmentations
        self.test_val_augment = test_val_augmentations
        self.trainGen = myDict()
        self.testGen = myDict()
        self.valGen = myDict()
        self.class_weights = myDict()
        self.output_range = output_range

        if class_mode in ['raw','multi-output']:
          self.labelMap = safe_label_to_labelmap_converter(self.labelCols)
        else:
          conv_labels = self.trainDF[self.labelCols[0]].unique()
          self.labelMap = safe_label_to_labelmap_converter(conv_labels)

        print('Label Map: {}\n labelCols: {}'.format(self.labelMap,self.labelCols))
        if isinstance(self.trainDF, pd.DataFrame) or isinstance(self.trainDF, str):
            self.trainGen.generator = ImageSequenceFromDF(dataFrame=self.trainDF, dataFolder=dataFolder, name ='train',
                 inputCol=inputCol, labelCols=labelCols, batch_size=batch_size, targetDim=targetDim, seed=seed,
                 color_mode=color_mode,shuffle=shuffle, class_mode=class_mode, augmentations=train_augmentations,
                 output_range=output_range, labelMap = self.labelMap
                 )
            self.trainGen = self._update_params(self.trainGen)
        
        if isinstance(self.testDF, pd.DataFrame) or isinstance(self.testDF, str):
            self.testGen.generator = ImageSequenceFromDF(dataFrame=self.testDF, dataFolder=dataFolder, name ='test',
                inputCol=inputCol, labelCols=labelCols, batch_size=batch_size, targetDim=targetDim, seed=seed,
                color_mode=color_mode,shuffle=False, class_mode=class_mode, augmentations=test_val_augmentations,
                output_range=output_range, labelMap = self.labelMap
                ) 
            self.testGen = self._update_params(self.testGen)
        if isinstance(self.valDF, pd.DataFrame) or isinstance(self.valDF, str):
            self.valGen.generator = ImageSequenceFromDF(dataFrame=self.valDF, dataFolder=dataFolder, name ='val',
                inputCol=inputCol, labelCols=labelCols, batch_size=batch_size, targetDim=targetDim, seed=seed,
                color_mode=color_mode,shuffle=False, class_mode=class_mode, augmentations=test_val_augmentations,
                output_range=output_range, labelMap = self.labelMap
                )  
            self.valGen = self._update_params(self.valGen)

    def _update_params(self, thisGen):
        thisGen.STEP_SIZE = thisGen.generator.STEP_SIZE
        if self.class_mode in ['binary', 'sparse', 'categorical']:
            thisGen.labelNames = safe_labelmap_converter(self.labelMap)
        elif self.class_mode in ['raw', 'multi_output']:
            thisGen.labelNames = self.labelCols
        thisGen.labelMap = self.labelMap
        return thisGen

    def load_generator(self):
        return self.trainGen, self.testGen, self.valGen

class ImageSequenceFromDF(Sequence):
    def __init__(self, dataFrame, dataFolder='', name ='train', labelMap = '',
                 inputCol="files", labelCols=['labels'], batch_size=16, 
                 targetDim=(96,96),color_mode="rgb",shuffle=True, seed=21, output_range=None,
                 class_mode="raw", augmentations=Compose([AUG.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
                 ):
        self.dataFolder,self.name = dataFolder, name
        self.inputCol, self.labelCols = inputCol, labelCols
        self.batch_size = batch_size
        self.targetDim, self.color_mode = targetDim, color_mode
        self.shuffle, self.seed = shuffle, seed
        self.class_mode = class_mode
        self.augment = augmentations
        self.output_range = output_range
        self.labelMap = labelMap

        self.convertCSVFile2DF(dataFrame)
        self._validateDF(self.dataFrame, self.name)
        self._dfConvertFilePath()
        self.N = self.dataFrame.shape[0]
        self.n = 0
        self.classes = labelCols
        
        self.STEP_SIZE = self.__len__()
        self.shuffleDataFrame = self.dataFrame
        
        if class_mode in ['raw', 'multi_output']:
            self.num_classes = len(list(labelCols))
        else:
            self.num_classes = len(self.labelMap.keys())
            self.dataFrame['convertedLabels']= self.dataFrame.apply(lambda x:self.convertLabelsFromLabelMap(x[labelCols[0]]), axis=1)
        
        self._add_info()

        if class_mode in ['raw', 'multi_output']:
            self._calculate_class_weights()
            
        self.on_epoch_end()

    def convertLabelsFromLabelMap(self,inlabel):
        if self.class_mode in ['sparse','categorical', 'binary']:
            outLabel = self.labelMap[str(inlabel)]
        return outLabel
            

    def img_processor(self, image):
        img = Image.open(image)
        if self.color_mode.upper() == 'RGB' and img.mode != 'RGB':
            img = img.convert('RGB')
        elif self.color_mode.upper() == 'RGBA'and img.mode != 'RGBA':
            img = img.convert('RGBA')
        elif self.color_mode.upper() == 'GRAYSCALE' and img.mode != 'L': 
            img = img.convert('L')
        img = img.resize((self.targetDim[0:2]),0)
        img = np.array(img, 'uint8')
        #img = np.array(img*(1./255)).astype('float32')
        return img

    def load_generator(self):
        return self.generator

    def _calculate_class_weights(self):
        pos_freq,neg_freq = compute_class_freqs(self.labels)
        wgtList=[]
        for x,y in zip(pos_freq,neg_freq):
            labelWgtDict = {0:x, 1:y}
            wgtList.append(labelWgtDict)
        self.class_weights = wgtList   

    def get_class_weights(self):
        return self.class_weights

    def __len__(self):
        return int(np.ceil(self.dataFrame.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx, batch = True):
        if batch:
            batchDF = self.shuffleDataFrame[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batchDF = self.shuffleDataFrame
        batch_x = batchDF[self.inputCol].to_numpy()
        
        if self.class_mode in ['binary', 'sparse', 'categorical']:
          batch_y =np.asarray(batchDF['convertedLabels'].to_list())
          #print('batch_y.shape {} {}'.format(batch_y.shape, batch_y[0].shape))
        else:
          batch_y = batchDF[self.labelCols].to_numpy()
        if self.augment != None:
            if self.output_range == None or isinstance(self.output_range, str):
                return np.stack(
                            [self.augment(image=self.img_processor(x))["image"] for x in batch_x],
                            axis=0), np.array(batch_y)
            else:
                return np.stack(
                            [np.clip(self.augment(image=self.img_processor(x))["image"],self.output_range[0],self.output_range[1]) for x in batch_x],
                            axis=0), np.array(batch_y)                
        else:
            return np.stack(
                        [self.img_processor(x) for x in batch_x],
                        axis=0), np.array(batch_y)          

    def __next__(self):
        batch_x, batch_y = self.__getitem__(self.n)
        self.n += 1
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0
        return batch_x, batch_y

    def __unbatch__(self):
        batch_x, batch_y = self.__getitem__(self.n, batch = False)
        self.on_epoch_end()
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle == True:
            self.shuffleDataFrame = self.dataFrame.sample(n=self.N, random_state=self.seed).reset_index(drop=True)
            self.seed+=1

    def _add_info(self):
        if self.color_mode.upper() == 'RGB':
            self.image_shape= self.targetDim+ (3,)
        elif self.color_mode.upper() == 'RGBA':
            self.image_shape= self.targetDim+ (4,)
        elif self.color_mode.upper() == 'GRAYSCALE':
            self.image_shape= self.targetDim+ (2,)
        if self.class_mode in ['raw', 'multi_output']:
            self.labels = self.dataFrame[self.labelCols].values
        else:
            self.labels = self.dataFrame[['convertedLabels']].values

    def check_imbalanced_dataset(self):
        dfC = self.dataFrame[self.labelCols].apply(pd.value_counts)
        self.dataSetStats = dataSetStats(dfC)
        return self.dataSetStats

    def convertCSVFile2DF(self,dataFrame):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame= dataFrame
        else:
            print('[INFO]: Reading CSV Files into DataFrame ', end='')
            self.dataFrame = pd.read_csv(dataFrame)
            print(' - Done!')

    def _validateDF(self,df, name):
            inPresent = True if self.inputCol in df.columns else False
            labelPresent = True if set(self.labelCols).issubset(df.columns) else False
            if inPresent and labelPresent:
                print('[INFO]: Dataframe {} Validation.. Success!'.format(name))
            else:
                print('[ERROR]: Dataframe {} Validation.. Failure!'.format(name)) 
                print('[---->]: Label Validation- {} : Input Validation - {}'.format(
                    'PASS' if labelPresent else 'FAIL','PASS' if inPresent else 'FAIL',))

    def _get_sample_full_path(self, fileName):
        return os.path.join(self.dataFolder, fileName)

    def _createFullInputPath(self, df):
        df[self.inputCol] = df[self.inputCol].map(lambda row : self._get_sample_full_path(row))
        return df

    def _dfConvertFilePath(self):
        self.dataFrame= self._createFullInputPath(self.dataFrame)

