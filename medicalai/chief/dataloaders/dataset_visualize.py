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
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.cm as cm

class dataSetStats(object):
    def __init__(self, dataFrame):
        super().__init__()
        self.dataFrame = dataFrame
        self.positve_class_count = dataFrame.loc[1,:]
        self.negative_class_count = dataFrame.loc[0,:]

    @staticmethod
    def _plot_hbar_df(pos, ax, figsize=(8,8),color='slateblue', 
                title = "Title", xaxis_title = "X_AXIS", label_y_offset= 0.45, 
                label_fontsize = 15):
        offsetFactor=len(pos.shape)
        maxVal = pos.max() if len(pos.shape)==1 else pos.max().max()
        pos.plot(kind='barh', figsize=figsize, color=color, alpha=0.5, ax = ax)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(xaxis_title, fontsize=18)
        for i in ax.patches:
            if maxVal - i.get_width()> maxVal*(1/10):
                ax.text(i.get_width()*(1+.03), i.get_y()+label_y_offset/offsetFactor, 
                        str(i.get_width()), fontsize=label_fontsize-offsetFactor,
                        color='blue')
            else:
                ax.text(i.get_width()-maxVal/10, i.get_y()+label_y_offset/offsetFactor, 
                        str(i.get_width()), fontsize=label_fontsize-offsetFactor,
                        color='black')

        ax.invert_yaxis()
        return ax

    def plot_samples_count(self, figsize=(8,8),color='red', 
                    title = "Class Distribution of Dataset",
                    xaxis_title = "No. of Samples per Class", 
                    label_y_offset= 0.45, label_fontsize = 15):
        fig, ax = plt.subplots()
        ax = self._plot_hbar_df(self.positve_class_count, ax,figsize=figsize,color=color, 
                    title = title, xaxis_title = xaxis_title, label_y_offset=label_y_offset,
                    label_fontsize=label_fontsize)
        plt.tight_layout(pad=2)
        return fig 

    def plot_samples_count(self, figsize=(8,8),color='red', showPlot= True,
                    title = "Class Count of Dataset",
                    xaxis_title = "No. of Samples per Class", 
                    label_y_offset= 0.45, label_fontsize = 15):
        fig, ax = plt.subplots()
        ax = self._plot_hbar_df(self.positve_class_count, ax,figsize=figsize,color=color, 
                    title = title, xaxis_title = xaxis_title, label_y_offset=label_y_offset,
                    label_fontsize=label_fontsize)
        plt.tight_layout(pad=2)
        if showPlot:
            plt.show()
        return fig        
       
    def plot_dataset_distribution(self,figsize=(8,10),color=['red', 'dodgerblue'], 
                    title = "Class Frequencies of Dataset", showPlot= True,
                    xaxis_title = "Negative vs Positive Sample Distribution", 
                    label_y_offset= 0.45, label_fontsize = 15):
        meanDF = self.dataFrame.copy().T

        meanDF['total']=meanDF.sum(axis=1)
        for x in self.dataFrame.index.values.tolist():
            meanDF[x] = meanDF[x]/meanDF['total']
        meanDF = meanDF.drop(['total'], axis = 1)
        meanDF = meanDF.apply(lambda x:round(x,3))
        meanDF.rename(columns ={0:'Negative',1:'Positive'}, inplace=True)
        self.sample_freq_df = meanDF
        fig, ax = plt.subplots()
        ax = self._plot_hbar_df(meanDF, ax,figsize=figsize,color=color, 
                    title = title, xaxis_title = xaxis_title, label_y_offset=label_y_offset,
                    label_fontsize=label_fontsize)
        plt.tight_layout(pad=2)
        if showPlot:
            plt.show()
        return fig

    def plot_all_dataset_analysis(self, returnPlot = False, showPlot= False):
        fig1 = self.plot_samples_count(showPlot=showPlot)
        fig2 = self.plot_dataset_distribution(showPlot=showPlot)
        if returnPlot:
            return [fig1, fig2]