import matplotlib.pyplot as plt
import numpy as np 
from tf_explain.core import ExtractActivations
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.smoothgrad import SmoothGrad


def predict_with_gradcam(model, imgNP, labels, selected_labels,
                    layer_name='bn' , expected = None):
    #preprocessed_input = load_image(img, image_dir, df)
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
            gradcam = explainer.explain((imgNP,labels),model,layer_name, class_index=i)
            plt.subplot(151 + j)
            plt.title(labels[i]+": p="+str(predictions[0][i]))
            plt.axis('off')
            plt.imshow(imgNP[0],cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1


