# medicalai.chief.model_metrics package

## Submodules

## medicalai.chief.model_metrics.modelstats module


### medicalai.chief.model_metrics.modelstats.bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000)

### medicalai.chief.model_metrics.modelstats.classify_report(y_true, y_pred)

### medicalai.chief.model_metrics.modelstats.compute_class_freqs(labels)
Compute positive and negative frequences for each class.


* **Parameters**

    **labels** (*np.array*) – matrix of labels, size (num_examples, num_classes)



* **Returns**

    array of positive frequences for each

        class, size (num_classes)

    negative_frequencies (np.array): array of negative frequences for each

        class, size (num_classes)




* **Return type**

    positive_frequencies (np.array)



### medicalai.chief.model_metrics.modelstats.confidence_intervals(class_labels, statistics)

### medicalai.chief.model_metrics.modelstats.false_negatives(expected, preds, threshold=0.5)
Count false positives.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **pred** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    false negatives



* **Return type**

    false_neg (int)



### medicalai.chief.model_metrics.modelstats.false_positives(expected, preds, threshold=0.5)
Count false positives.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **preds** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    false positives



* **Return type**

    false_pos (int)



### medicalai.chief.model_metrics.modelstats.generate_evaluation_report(CLASS_NAMES, predictions, groundTruth=None, generator=None, returnPlot=True, showPlot=True, printStat=True, \*\*kwargs)
Generates Evaluation PDF Report for a Test/Validation experimentation. Ground truth needs to be passed to generate the pdf report.


* **Parameters**

    
    * **CLASS_NAMES** (*list*) – List of Label names or class names of dataset.


    * **predictions** (*np.array*) – Predicted output of test data.


    * **groundTruth** (*np.array*) – Ground truth of test data.


    * **generator** (*Optional*) – If generator method used in training, pass the generator.


    * **returnPlot** (*Bool*) – Returns the plot handle if set to True


    * **showPlot** (*Bool*) – Display the plot if set to True. [IMP: Until the plot is closed, the code execution is blocked.]


    * **printStat** (*Bool*) – Print the statistics of the experiment on the console if set to True. T


    * **\*\*kwargs** (*Optional*) – Plot Setting Arguments



* **Returns**

    true positives



* **Return type**

    true_pos (int)



### medicalai.chief.model_metrics.modelstats.get_accuracy(expected, preds, threshold=0.9)
Compute accuracy of predictions at threshold.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **preds** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    accuracy of predictions at threshold



* **Return type**

    accuracy (float)



### medicalai.chief.model_metrics.modelstats.get_accuracy_score(test_labels, test_predictions)

### medicalai.chief.model_metrics.modelstats.get_curve(gt, pred, target_names, curve='roc', returnPlot=False, showPlot=True, axes=None, \*\*kwargs)

### medicalai.chief.model_metrics.modelstats.get_false_neg(y, pred, th=0.5)

### medicalai.chief.model_metrics.modelstats.get_false_pos(y, pred, th=0.5)

### medicalai.chief.model_metrics.modelstats.get_npv(expected, preds, threshold=0.5)
Compute NPV of predictions at threshold.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **preds** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    negative predictive value of predictions at threshold



* **Return type**

    NPV (float)



### medicalai.chief.model_metrics.modelstats.get_performance_metrics(y, pred, class_labels, tp=<function get_true_pos>, tn=<function get_true_neg>, fp=<function get_false_pos>, fn=<function get_false_neg>, acc=None, prevalence=None, spec=None, sens=None, ppv=None, npv=None, auc=None, f1=None, thresholds=[])

### medicalai.chief.model_metrics.modelstats.get_ppv(expected, preds, threshold=0.5)
Compute PPV of predictions at threshold.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **preds** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    positive predictive value of predictions at threshold



* **Return type**

    PPV (float)



### medicalai.chief.model_metrics.modelstats.get_prevalence(expected)
Compute accuracy of predictions at threshold.


* **Parameters**

    **expected** (*np.array*) – ground truth, size (n_examples)



* **Returns**

    prevalence of positive cases



* **Return type**

    prevalence (float)



### medicalai.chief.model_metrics.modelstats.get_roc_curve(labels, predicted_vals, groundTruth=None, generator=None, returnPlot=False, showPlot=True, axes=None, \*\*kwargs)

### medicalai.chief.model_metrics.modelstats.get_sensitivity(expected, preds, threshold=0.5)
Compute sensitivity of predictions at threshold.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **preds** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    probability that our test outputs positive given that the case is actually positive



* **Return type**

    sensitivity (float)



### medicalai.chief.model_metrics.modelstats.get_specificity(expected, preds, threshold=0.5)
Compute specificity of predictions at threshold.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **preds** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    probability that the test outputs negative given that the case is actually negative



* **Return type**

    specificity (float)



### medicalai.chief.model_metrics.modelstats.get_true_neg(y, pred, th=0.5)

### medicalai.chief.model_metrics.modelstats.get_true_pos(y, pred, th=0.5)

### medicalai.chief.model_metrics.modelstats.get_weighted_loss(pos_weights, neg_weights, epsilon=1e-07)
Return weighted loss function given negative weights and positive weights.


* **Parameters**

    
    * **pos_weights** (*np.array*) – array of positive weights for each class, size (num_classes)


    * **neg_weights** (*np.array*) – array of negative weights for each class, size (num_classes)



* **Returns**

    weighted loss function



* **Return type**

    weighted_loss (function)



### medicalai.chief.model_metrics.modelstats.model_performance_metrics(y, pred, class_labels, tp=<function get_true_pos>, tn=<function get_true_neg>, fp=<function get_false_pos>, fn=<function get_false_neg>, thresholds=[])

### medicalai.chief.model_metrics.modelstats.platt_scaling(y, pred, class_labels)

### medicalai.chief.model_metrics.modelstats.plot_calibration_curve(y, pred, class_labels)

### medicalai.chief.model_metrics.modelstats.plot_confusion_matrix(model=None, test_data=None, test_labels=None, labelNames=None, title='Confusion Matrix', predictions=None, showPlot=True, returnPlot=False)

### medicalai.chief.model_metrics.modelstats.print_classification_report(y_true, y_pred)

### medicalai.chief.model_metrics.modelstats.print_cohen_kappa_score(y_true, y_pred)

### medicalai.chief.model_metrics.modelstats.render_df_as_table(data, title='Table', col_width=3.0, row_height=0.625, font_size=18, header_color='#655EE5', row_colors=['#f1f1f2', 'w'], edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, resetIndex=False, ax=None, \*\*kwargs)

### medicalai.chief.model_metrics.modelstats.true_negatives(expected, preds, threshold=0.5)
Count true negatives.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **preds** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    true negatives



* **Return type**

    true_neg (int)



### medicalai.chief.model_metrics.modelstats.true_positives(expected, preds, threshold=0.5)
Count true positives.


* **Parameters**

    
    * **expected** (*np.array*) – ground truth, size (n_examples)


    * **preds** (*np.array*) – model output, size (n_examples)


    * **threshold** (*float*) – cutoff value for positive prediction from model



* **Returns**

    true positives



* **Return type**

    true_pos (int)


## Module contents
