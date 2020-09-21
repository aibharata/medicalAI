from __future__ import absolute_import
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import os
class AUROC_Callback(Callback):
    '''
    Sample Usage: 
    auroc = AUROC_Callback(
            generator=generator,
            workers=generator_workers,
        ) 
    '''
    def __init__(self, generator, workers=1):
        super().__init__()
        self.generator = generator
        self.workers = workers

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.generator, workers=self.workers)
        y_true= self.generator.labels
        meanAUROC = roc_auc_score(y_true,y_pred)
        print(' - mAUROC:', meanAUROC)

