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
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score,roc_curve,f1_score
import matplotlib.pyplot as plt
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



def get_roc_curve(labels, predicted_vals, groundTruth= None, generator=None , returnPlot = False, showPlot= True):
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
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                "Error in generating ROC curve for",labels[i],
                "\nDataset lacks enough examples."
            )
    if showPlot==True:
        plt.show()
    if returnPlot==True:
        return auc_roc_vals, plt
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


def get_curve(gt, pred, target_names, curve='roc',returnPlot = False, showPlot= True):
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt[:, i], pred[:, i])
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            a, b, _ = curve_function(gt[:, i], pred[:, i])
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
        elif curve == 'precision_recall_curve':
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            average_precision = average_precision_score(gt[:, i], pred[:, i])
            label = target_names[i] + " Avg.: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where='post', label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
    if showPlot==True:
        plt.show()
    if returnPlot==True:
        return plt

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


def get_detailed_evaluation(CLASS_NAMES, predictions, groundTruth=None,  generator=None ,):
    gt_one_hot_vec = np.identity(OUTPUT_CLASSES)[groundTruth[:,0]]
    OUTPUT_CLASSES = len(CLASS_NAMES)
    if groundTruth is not None and generator is None:
        auc_roc_vals, roc_plt = get_roc_curve(CLASS_NAMES,predictions,groundTruth=gt_one_hot_vec, returnPlot = True, showPlot= True)
    elif generator is not None:
        auc_roc_vals, roc_plt = get_roc_curve(CLASS_NAMES,predictions,generator=generator, returnPlot = True, showPlot= True)

    prefMetrics = get_performance_metrics(gt_one_hot_vec, predictions, CLASS_NAMES, acc=get_accuracy, prevalence=get_prevalence, 
                            sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score, f1=f1_score)
    print(prefMetrics)
    statistics = bootstrap_auc(gt_one_hot_vec, predictions, CLASS_NAMES)
    confInterval = confidence_intervals(CLASS_NAMES, statistics)
    print(confInterval)

    prc_plt = get_curve(gt_one_hot_vec, predictions, CLASS_NAMES, curve='precision_recall_curve', returnPlot = False, showPlot= True)

    plots =[roc_plt, prc_plt]
    allMetrics = [prefMetrics, confInterval, auc_roc_vals]
    return allMetrics, plots