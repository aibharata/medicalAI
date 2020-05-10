import matplotlib.pyplot as plt
import numpy as np 
from tf_explain.core import ExtractActivations
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.smoothgrad import SmoothGrad

'''
def _gradcam(imgNP,  labels, selected_labels,
                    layer_name='bn' , expected = None, predictions =None):
'''
def predict_with_gradcam(model, imgNP,  labels, selected_labels,
                    layer_name='bn' , expected = None, predictions =None, showPlot=False):
    '''
    TODO: Explainer requires model to sent for every call. This may be expensive.
          Need to find a way to Initialize the model or share with predict engine. 
          Else the memory required may double.
    '''
    #preprocessed_input = load_image(img, image_dir, df)
    if predictions is None:
        predictions = model.predict(imgNP)
    
    #print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title(("Original - " + expected) if expected is not None else "Original")
    plt.axis('off')
    plt.imshow(imgNP[0], cmap='gray')
    
    explainer = GradCAM()
    
    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            #print("Generating gradcam for class",labels[i])
            #gradcam = grad_cam(model, imgNP, i, layer_name)
            #print("the class index is :", i)
            gradcam = explainer.explain(validation_data=(imgNP,labels),model= model,layer_name= layer_name, class_index=i)
            plt.subplot(151 + j)
            #plt.title(labels[i]+": p="+str(predictions[0][i]))
            plt.title("{:}: p={:.2f}%".format(labels[i],predictions[0][i]*100))
            plt.axis('off')
            plt.imshow(imgNP[0],cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1
    if showPlot:
        plt.show()

