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
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score,roc_curve,f1_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid',palette="muted", color_codes=True)
import six
from matplotlib.backends.backend_pdf import PdfPages
def true_positives(expected, preds, threshold=0.5):
    """
    Count true positives.

    Args:

        expected (np.array): ground truth, size (n_examples)
        preds (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        true_pos (int): true positives
    """
    true_pos = 0
    thresholded_preds = preds >= threshold
    true_pos = np.sum((expected == 1) & (thresholded_preds == 1))
    return true_pos

def true_negatives(expected, preds, threshold=0.5):
    """
    Count true negatives.

    Args:

        expected (np.array): ground truth, size (n_examples)
        preds (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        true_neg (int): true negatives
    """
    true_neg = 0
    thresholded_preds = preds >= threshold
    true_neg = np.sum((expected == 0) & (thresholded_preds == 0))
    return true_neg

def false_positives(expected, preds, threshold=0.5):
    """
    Count false positives.

    Args:

        expected (np.array): ground truth, size (n_examples)
        preds (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        false_pos (int): false positives
    """
    false_pos = 0
    thresholded_preds = preds >= threshold
    false_pos = np.sum((expected == 0) & (thresholded_preds == 1))
    return false_pos

def false_negatives(expected, preds, threshold=0.5):
    """
    Count false positives.

    Args:

        expected (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        false_neg (int): false negatives
    """
    false_neg = 0
    thresholded_preds = preds >= threshold
    false_neg = np.sum((expected == 1) & (thresholded_preds == 0))
    return false_neg

def get_accuracy(expected, preds, threshold=0.9):
    """
    Compute accuracy of predictions at threshold.

    Args:

        expected (np.array): ground truth, size (n_examples)
        preds (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        accuracy (float): accuracy of predictions at threshold
    """
    accuracy = 0.0
    TP = true_positives(expected, preds, threshold)
    FP = false_positives(expected, preds, threshold)
    TN =true_negatives(expected, preds, threshold)
    FN =false_negatives(expected, preds, threshold)
    accuracy = (TP + TN)/(TP + FP + TN + FN)
    
    return accuracy

def get_prevalence(expected):
    """
    Compute accuracy of predictions at threshold.

    Args:

        expected (np.array): ground truth, size (n_examples)

    Returns:

        prevalence (float): prevalence of positive cases
    """
    prevalence = 0.0
    prevalence = np.mean(expected)
    return prevalence

def get_sensitivity(expected, preds, threshold=0.5):
    """
    Compute sensitivity of predictions at threshold.

    Args:

        expected (np.array): ground truth, size (n_examples)
        preds (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        sensitivity (float): probability that our test outputs positive given that the case is actually positive
    """
    sensitivity = 0.0
    TP =true_positives(expected, preds, threshold)
    FN =false_negatives(expected, preds, threshold)
    sensitivity = TP/(TP + FN)
    return sensitivity

def get_specificity(expected, preds, threshold=0.5):
    """
    Compute specificity of predictions at threshold.

    Args:

        expected (np.array): ground truth, size (n_examples)
        preds (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        specificity (float): probability that the test outputs negative given that the case is actually negative
    """
    specificity = 0.0
    TN = true_negatives(expected, preds, threshold)
    FP = false_positives(expected , preds, threshold)
    specificity = TN/(TN + FP)
    return specificity

def get_ppv(expected, preds, threshold=0.5):
    """
    Compute PPV of predictions at threshold.

    Args:

        expected (np.array): ground truth, size (n_examples)
        preds (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        PPV (float): positive predictive value of predictions at threshold
    """
    PPV = 0.0
    TP = true_positives(expected, preds, threshold)
    FP = false_positives(expected, preds, threshold)
    PPV = TP/(TP+FP)
    return PPV

def get_npv(expected, preds, threshold=0.5):
    """
    Compute NPV of predictions at threshold.

    Args:

        expected (np.array): ground truth, size (n_examples)
        preds (np.array): model output, size (n_examples)
        threshold (float): cutoff value for positive prediction from model

    Returns:

        NPV (float): negative predictive value of predictions at threshold
    """
    NPV = 0.0
    TN = true_negatives(expected, preds, threshold)
    FN = false_negatives(expected, preds, threshold)
    NPV = TN/(TN+FN)
    return NPV

def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:

        labels (np.array): matrix of labels, size (num_examples, num_classes)

    Returns:

        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)

    """
    N = labels.shape[0] #np.sum(labels,axis=1)
    positive_frequencies = np.sum(labels == 1,axis=0) / N
    negative_frequencies = np.sum(labels == 0,axis=0) / N
    return positive_frequencies, negative_frequencies

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:

      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:

      weighted_loss (function): weighted loss function
    """
    from tensorflow.keras import backend as K
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:

            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:

            loss (Tensor): overall scalar loss summed across all classes
        """
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += -1*(K.mean(((neg_weights[i] * (1 - y_true[:,i]) * K.log(1 - y_pred[:,i] + epsilon)) + pos_weights[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon)), axis = 0))
        return loss 
    return weighted_loss

def get_curve(gt, pred, target_names, curve='roc',returnPlot = False, showPlot= True, axes=None, **kwargs):
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt[:, i], pred[:, i])
            label = target_names[i] + " AUC: %.3f " % auc_roc
            a, b, _ = curve_function(gt[:, i], pred[:, i])
            if showPlot:
                fig = plt.figure(2, figsize=(10, 10))
                ax = plt
                ax.xlabel("False positive rate")
                ax.ylabel("True positive rate")
                ax.title('ROC Curve')
            else:
                ax = axes
                ax.set_xlabel('False positive rate')
                ax.set_ylabel('True positive rate')
                ax.set_title('ROC Curve')                
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
        elif curve == 'precision_recall_curve':
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            average_precision = average_precision_score(gt[:, i], pred[:, i])
            label = target_names[i] + " Avg.: %.3f " % average_precision
            if showPlot:
                fig = plt.figure(2, figsize=(10, 10))
                ax = plt
                ax.xlabel('Recall')
                ax.ylabel('Precision')
                ax.title('Precision Recall Curve',fontsize = 18, color='r')
                ax.ylim([0.0, 1.05])
                ax.xlim([0.0, 1.0])
            else:
                ax = axes
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision Recall Curve',fontsize = 18, color='r')  
                ax.set_ylim([0.0, 1.05])
                ax.set_xlim([0.0, 1.0])
            plt.step(recall, precision, where='post', label=label)

        plt.legend(loc='best')
    #print('showPlot ',showPlot)
    if showPlot==True:
        plt.show()
        if returnPlot==True:
            return fig
    else:
        if returnPlot==True:
            return ax



def get_roc_curve(labels, predicted_vals, groundTruth= None, generator=None , returnPlot = False, showPlot= True, axes=None, **kwargs):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            if generator is not None and groundTruth is None:
                gt = generator.labels[:, i]
            elif groundTruth is not None and generator is None:
                gt = groundTruth[:, i]
            else:
                print('Wrong Configuration: Only groundTruth or generator can be set- Not BOTH')
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            if showPlot:
                fig = plt.figure(1,figsize=(10, 10))
                ax = plt
                ax.xlabel('False positive rate')
                ax.ylabel('True positive rate')
                ax.title('ROC Curve',fontsize = 18, color='r')
            else:
                ax = axes
                ax.set_xlabel('False positive rate')
                ax.set_ylabel('True positive rate')
                ax.set_title('ROC Curve',fontsize = 18, color='r')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.plot(fpr_rf, tpr_rf,
                        label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")

            
            ax.legend(loc='best')
        except Exception as err:
            print("[ERROR]: in generating ROC curve for",labels[i], '\n',err)
    #print('showPlot ROC Curve',showPlot)
    if showPlot==True:
        plt.show()
        if returnPlot==True:
            return auc_roc_vals, fig
        else:
            return auc_roc_vals
    else:
        if returnPlot==True:
            return auc_roc_vals, ax
        else:
            return auc_roc_vals        


def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics


def get_true_pos(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))


def get_true_neg(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))


def get_false_neg(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))


def get_false_pos(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))

def get_performance_metrics(y, pred, class_labels, tp=get_true_pos,
                            tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg,
                            acc=None, prevalence=None, spec=None,
                            sens=None, ppv=None, npv=None, auc=None, f1=None,
                            thresholds=[]):

    if len(thresholds) != len(class_labels):
        thresholds = [.5] * len(class_labels)

    columns = ["", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
               "Sensitivity",
               "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = class_labels[i]
        df.loc[i][1] = round(tp(y[:, i], pred[:, i]),
                             3) if tp != None else "Not Defined"
        df.loc[i][2] = round(tn(y[:, i], pred[:, i]),
                             3) if tn != None else "Not Defined"
        df.loc[i][3] = round(fp(y[:, i], pred[:, i]),
                             3) if fp != None else "Not Defined"
        df.loc[i][4] = round(fn(y[:, i], pred[:, i]),
                             3) if fn != None else "Not Defined"
        df.loc[i][5] = round(acc(y[:, i], pred[:, i], thresholds[i]),
                             3) if acc != None else "Not Defined"
        df.loc[i][6] = round(prevalence(y[:, i]),
                             3) if prevalence != None else "Not Defined"
        df.loc[i][7] = round(sens(y[:, i], pred[:, i], thresholds[i]),
                             3) if sens != None else "Not Defined"
        df.loc[i][8] = round(spec(y[:, i], pred[:, i], thresholds[i]),
                             3) if spec != None else "Not Defined"
        df.loc[i][9] = round(ppv(y[:, i], pred[:, i], thresholds[i]),
                             3) if ppv != None else "Not Defined"
        df.loc[i][10] = round(npv(y[:, i], pred[:, i], thresholds[i]),
                              3) if npv != None else "Not Defined"
        df.loc[i][11] = round(auc(y[:, i], pred[:, i]),
                              3) if auc != None else "Not Defined"
        df.loc[i][12] = round(f1(y[:, i], pred[:, i] > thresholds[i]),
                              3) if f1 != None else "Not Defined"
        df.loc[i][13] = round(thresholds[i], 3)

    df = df.set_index("")
    return df

def model_performance_metrics(y, pred, class_labels, tp=get_true_pos,
                            tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg,
                            thresholds=[]):
    return get_performance_metrics(y, pred, class_labels, tp=get_true_pos,
                            tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg,acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score, f1=f1_score,
                            thresholds=[])

def confidence_intervals(class_labels, statistics):
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df



from sklearn.calibration import calibration_curve
def plot_calibration_curve(y, pred,class_labels):
    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()


from sklearn.linear_model import LogisticRegression as LR 

def platt_scaling(y, pred,class_labels):
    y_train = y
    pred_train = pred
    pred_calibrated = np.zeros_like(pred)

    for i in range(len(class_labels)):
        lr = LR(solver='liblinear', max_iter=10000)
        lr.fit(pred_train[:, i].reshape(-1, 1), y_train[:, i])    
        pred_calibrated[:, i] = lr.predict_proba(pred[:, i].reshape(-1, 1))[:,1]
    return pred_calibrated


def generate_evaluation_report(CLASS_NAMES, predictions, groundTruth=None,  generator=None, returnPlot = True, showPlot= True , printStat=True, **kwargs):
    """
    Generates Evaluation PDF Report for a Test/Validation experimentation. Ground truth needs to be passed to generate the pdf report.

    Args:

        CLASS_NAMES (list): List of Label names or class names of dataset. 
        predictions (np.array): Predicted output of test data.
        groundTruth (np.array): Ground truth of test data.
        generator (Optional): If generator method used in training, pass the generator.
        returnPlot (Bool): Returns the plot handle if set to `True`
        showPlot (Bool): Display the plot if set to `True`. [IMP: Until the plot is closed, the code execution is blocked.]
        printStat (Bool): Print the statistics of the experiment on the console if set to `True`. T
        **kwargs (Optional): Plot Setting Arguments 

    Returns:

        true_pos (int): true positives
    """    
    OUTPUT_CLASSES = len(CLASS_NAMES)
    if groundTruth is not None and generator is None:
        gt_one_hot_vec = np.identity(OUTPUT_CLASSES)[groundTruth[:,0]]
        gt = groundTruth[:,0]
    elif generator is not None:
        gt_one_hot_vec = np.identity(OUTPUT_CLASSES)[generator.labels]
        gt = generator.labels

    if not showPlot:
        nrows=kwargs['nrows'] if 'nrows' in kwargs else 2
        ncols=kwargs['ncols'] if 'ncols' in kwargs else 1
        pad = kwargs['pad'] if 'pad' in kwargs else 10
        hspace = kwargs['hspace'] if 'hspace' in kwargs else 0.3
        hspace2 = kwargs['hspace2'] if 'hspace2' in kwargs else 0.3
        figSize = kwargs['figsize'] if 'figsize' in kwargs else (10,10)
        modelName = kwargs['modelName'] if 'modelName' in kwargs else 'Model Name'
        pdfName = kwargs['pdfName'] if 'pdfName' in kwargs and kwargs['pdfName'] is not None else modelName
        fig = plt.figure(figsize=figSize)    
        axs = fig.subplots(nrows=nrows, ncols=ncols)
        fig.tight_layout(pad=pad)
        plt.subplots_adjust(hspace=hspace)
    else:
        axs = [None for x in range(0,nrows*ncols)]
    auc_roc_vals, roc_plt = get_roc_curve(CLASS_NAMES,predictions,groundTruth=gt_one_hot_vec, generator=None,
                                            returnPlot = True, showPlot= showPlot, axes=axs[0])
    prefMetrics = get_performance_metrics(gt_one_hot_vec, predictions, CLASS_NAMES, acc=get_accuracy, prevalence=get_prevalence, 
                            sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score, f1=f1_score)
    
    statistics = bootstrap_auc(gt_one_hot_vec, predictions, CLASS_NAMES)
    confInterval = confidence_intervals(CLASS_NAMES, statistics)


    prc_plt = get_curve(gt_one_hot_vec, predictions, CLASS_NAMES, curve='precision_recall_curve', returnPlot = False, showPlot= showPlot, axes=axs[1])

    
    dfSpec = prefMetrics[['Sensitivity','Specificity', 'PPV', 'NPV', 'AUC', 'F1']]#.astype(float)
    dfAcc= prefMetrics[['TP', 'TN', 'FP', 'FN', 'Accuracy', 'Prevalence']]

    fig2 = plt.figure(figsize=(15,15))
    fig2.suptitle("AI-Bharata MedicalAI - Model Evaluation Report Generator\n\nModel Name : {:}\n".format(modelName.replace('\\','/').split('/')[-1]), horizontalalignment='center', fontsize=18,color='grey') 
    axs2 = fig2.subplots(nrows=3, ncols=1)
    axs2[0] = render_df_as_table(dfSpec,  ax= axs2[0], title = 'Model Sensitivity and Specificity Details (Th={})'.format(prefMetrics['Threshold'][0]), header_columns=0, col_width=2.2, resetIndex=True)
    axs2[1] = render_df_as_table(dfAcc, ax= axs2[1], title = 'Validation Accuracy Details (Th={})'.format(prefMetrics['Threshold'][0]),header_columns=0, col_width=2.2, header_color='#DE2E81', resetIndex=True)
    axs2[2] = render_df_as_table(confInterval, ax= axs2[2],title = 'Confidence Interval',header_columns=0, col_width=4, header_color='#2BC4C5', resetIndex=True)

    fig3 = plt.figure(figsize=(10,10))
    axs3 = fig3.subplots(nrows=1, ncols=1)
    con_mat_norm_df, con_mat_df,Accuracy,cohenKappaScore =  _CM_calculate(np.argmax(predictions, axis=-1),gt,CLASS_NAMES)
    axs3 = _Plot_Heatmap_from_DF(con_mat_norm_df, title="Confusion Matrix Normalized", ax = axs3, Accuracy=Accuracy, cohenKappaScore=cohenKappaScore) 
    #axs3[1] = _Plot_Heatmap_from_DF(con_mat_df, title="Confusion Matrix", ax = axs3[1], Accuracy=Accuracy, cohenKappaScore=cohenKappaScore) 
    #fig3.tight_layout()
    allMetrics = [prefMetrics, confInterval, auc_roc_vals]
    if showPlot:
        plots =[roc_plt, prc_plt,axs2[0], axs2[1], axs2[2], axs3[0],axs3[1] ]
    else:
        plots =[fig2,fig, fig3]
    if printStat==True:
        print(dfSpec)
        print(dfAcc)
        print(confInterval)

    with PdfPages(pdfName+'_report.pdf') as pdf:
        for x in plots:
            pdf.savefig(x)

    if returnPlot:
        return allMetrics, plots


def render_df_as_table(data, title = 'Table', col_width=3.0, row_height=0.625, font_size=18,
                    header_color='#655EE5', row_colors=['#f1f1f2', 'w'], edge_color='w',
                    bbox=[0, 0, 1, 1], header_columns=0, resetIndex=False,
                    ax=None, **kwargs):
    if resetIndex:
        data = data.reset_index()
    try:
        data =data.rename(columns={'index': 'CLASSES'})
    except:
        v = None
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
        plt.title(title, fontsize = font_size+2)
    else:
        ax.axis('off')
        ax.set_title(title, fontsize = font_size+2)
    df_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    df_table.auto_set_font_size(False)
    df_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(df_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

from sklearn.metrics import accuracy_score,classification_report,cohen_kappa_score

def get_accuracy_score(test_labels, test_predictions):
	return accuracy_score(test_labels, test_predictions)

def classify_report(y_true,y_pred):
	return(classification_report(y_true, y_pred, digits=3))

def print_classification_report(y_true,y_pred):
	print(classification_report(y_true, y_pred, digits=3))

def print_cohen_kappa_score(y_true,y_pred):
	print(cohen_kappa_score(y_true, y_pred, digits=3))

def _CM_calculate(predictions,ground_truth,classNames):
    con_mat = confusion_matrix(ground_truth,predictions)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    cohenKappaScore = cohen_kappa_score(ground_truth,predictions)
    Accuracy =float(get_accuracy_score(ground_truth, predictions)*100)
    con_mat_norm_df = pd.DataFrame(con_mat_norm,
                                index = classNames, 
                                columns = classNames)
    con_mat_df = pd.DataFrame(con_mat,
                                index = classNames, 
                                columns = classNames)
    return con_mat_norm_df, con_mat_df,Accuracy,cohenKappaScore

def _Plot_Heatmap_from_DF(pdDf, title="Confusion Matrix", ax = None, Accuracy=None, cohenKappaScore=None, printTitle=False):
    subTitleStr= '➤ Model Accuracy:{:.2f}% '.format(Accuracy) if Accuracy is not None else ""
    subTitleStr+='➤ Cohen Kappa Score {:.3f}'.format(cohenKappaScore) if cohenKappaScore is not None else ""
    if ax is None:
        fig,ax = plt.figure(figsize=(10, 10))
    sns.heatmap(pdDf, annot=True, cmap=plt.get_cmap('PuRd') , ax =ax)
    if printTitle:
        ax.set_title("AI-Bharata MedicalAI\n\n", loc='center', fontsize=18,color='grey')
        ax.set_title('{:}\n{}'.format(title,subTitleStr),loc='left', fontsize=13, )
    else:
        ax.set_title("{:}\n\n".format(title), loc='center', fontsize=18,color='grey')
        ax.set_title('{}'.format(subTitleStr),loc='left', fontsize=13, )
    ax.set_ylabel('True label',fontsize=18)
    ax.set_xlabel('Predicted label',fontsize=18)
    return ax


def plot_confusion_matrix(model=None, test_data=None, test_labels =None, labelNames=None, title='Confusion Matrix', predictions=None, showPlot=True, returnPlot=False):
	if predictions is None:
		test_predictions = np.argmax(model.predict(test_data), axis=-1)
	else:
		test_predictions = np.argmax(predictions, axis=-1)
	#print(classify_report(test_labels,test_predictions))
	cohenKappaScore = cohen_kappa_score(test_labels,test_predictions)
	#print('Cohen kappa Score:', cohenKappaScore)
	con_mat = confusion_matrix(test_labels,test_predictions)
	con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
	
	con_mat_norm_df = pd.DataFrame(con_mat_norm,
					 index = labelNames, 
					 columns = labelNames)

	figure = plt.figure(figsize=(10, 10))
	sns.heatmap(con_mat_norm_df, annot=True, cmap=plt.get_cmap('PuRd') )

	plt.title("AI-Bharata MedicalAI\n\n", loc='center', fontsize=18,color='grey')
	
	plt.title('{:}\n➤ Model Accuracy:{:.2f}% ➤ Cohen Kappa Score {:.3f}'.format(title,float(get_accuracy_score(test_labels, test_predictions)*100),cohenKappaScore),loc='left', fontsize=13, )
	plt.ylabel('True label',fontsize=18)
	plt.xlabel('Predicted label',fontsize=18)
	if showPlot:
		plt.show()
	if returnPlot:
		return test_predictions,figure
	else:
		return test_predictions
