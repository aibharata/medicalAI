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
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
import os
import albumentations.augmentations.transforms as AUG
from albumentations import Compose
from .data_utils import *
from ..dataset_analysis import compute_class_freqs
from .dataset_visualize import *

import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE
class ImagePipelineFromDF(object):
    def __init__(self, dataFrame, dataFolder='',
                 inputCol="files", labelCols=['labels'], batch_size=16, 
                 targetDim=(96,96),augmentations=None, color_mode="rgb",
                 class_mode="raw", shuffle=True, seed=21, normalize = True, 
                 shuffle_buffer_size = 1000, cache= False, prefetch = True,
                 name = 'train'
                 ):
        self.convertCSVFile2DF(dataFrame)
        self.dataFolder = dataFolder
        self.inputCol, self.labelCols = inputCol, labelCols
        self.BATCH_SIZE = batch_size
        self.targetDim = targetDim
        self.augmentations = augmentations
        self.color_mode = color_mode
        self.class_mode = class_mode
        self.shuffle_buffer_size, self.shuffle = shuffle_buffer_size, shuffle
        self.normalize = normalize
        self.seed = seed
        self.prefetch,self.cache = prefetch,cache
        
        self._validateDF(self.dataFrame, name)
        self._dfConvertFilePath()
        self.N = self.dataFrame.shape[0]
        self.n = 0
        self.classes = labelCols
        self.num_classes = len(list(labelCols))
        self.STEP_SIZE = self.__len__()
        self.list_ds =tf.data.Dataset.from_tensor_slices((self.dataFrame[self.inputCol].values, 
                                                   self.dataFrame[self.labelCols].values))
        self.process_dataset()

        if name.lower=='train':
            repeat = True
            augment = True
        else:
            repeat = False
            augment = False

        self.generator = self.prepare_for_training(self.labeled_ds, repeat =repeat, augment = augment)

    def tfDataset(self):
        return self.generator

    @tf.function
    def decode_img(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        if self.normalize:
            img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, list(self.targetDim[0:2]))

    def process_path(self,file_path, labels):
        label = labels #self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        #img = self.augment(img,label)
        return img, label  

    def process_dataset(self):
        self.labeled_ds = self.list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)

    def prepare_for_training(self, ds, repeat =True, augment = True):
        """
        - If using is a small dataset, only load it once, and keep it in memory.
        - use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
        """
        if self.cache:
            if isinstance(self.cache, str):
                ds = ds.cache(self.cache)
            else:
                ds = ds.cache()
                
        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)

        if repeat:
            ds = ds.repeat()

        ds = ds.batch(self.BATCH_SIZE)
        if augment:
            ds = ds.map(self.augment,num_parallel_calls=AUTOTUNE)

        if self.prefetch:
            ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    #@tf.function
    def augment(self, image,label):
        image = tf.numpy_function(func=self.custom_augment, inp=[image], Tout=tf.float32)
        image = tf.clip_by_value(image, 0, 1)
        return image, label

    #@tf.function
    def custom_augment(self,image):
        image = np.uint8((image)*255)
        images  = np.stack([self.augmentations(image=x)["image"] for x in image], axis=0)
        return images

    def load_generator(self):
        return self.generator

    def __len__(self):
        return int(np.ceil(self.dataFrame.shape[0] / float(self.BATCH_SIZE)))

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
        df[self.inputCol] = df[self.inputCol].apply(lambda row : self._get_sample_full_path(row))
        return df

    def _dfConvertFilePath(self):
        self.dataFrame= self._createFullInputPath(self.dataFrame)