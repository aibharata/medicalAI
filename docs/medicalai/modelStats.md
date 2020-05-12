
# medicalai.chief.model_metrics.modelstats


## true_positives
```python
true_positives(expected, preds, threshold=0.5)
```

Count true positives.

Args:

    expected (np.array): ground truth, size (n_examples)
    preds (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    true_pos (int): true positives


## true_negatives
```python
true_negatives(expected, preds, threshold=0.5)
```

Count true negatives.

Args:

    expected (np.array): ground truth, size (n_examples)
    preds (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    true_neg (int): true negatives


## false_positives
```python
false_positives(expected, preds, threshold=0.5)
```

Count false positives.

Args:

    expected (np.array): ground truth, size (n_examples)
    preds (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    false_pos (int): false positives


## false_negatives
```python
false_negatives(expected, preds, threshold=0.5)
```

Count false positives.

Args:

    expected (np.array): ground truth, size (n_examples)
    pred (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    false_neg (int): false negatives


## get_accuracy
```python
get_accuracy(expected, preds, threshold=0.9)
```

Compute accuracy of predictions at threshold.

Args:

    expected (np.array): ground truth, size (n_examples)
    preds (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    accuracy (float): accuracy of predictions at threshold


## get_prevalence
```python
get_prevalence(expected)
```

Compute accuracy of predictions at threshold.

Args:

    expected (np.array): ground truth, size (n_examples)

Returns:

    prevalence (float): prevalence of positive cases


## get_sensitivity
```python
get_sensitivity(expected, preds, threshold=0.5)
```

Compute sensitivity of predictions at threshold.

Args:

    expected (np.array): ground truth, size (n_examples)
    preds (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    sensitivity (float): probability that our test outputs positive given that the case is actually positive


## get_specificity
```python
get_specificity(expected, preds, threshold=0.5)
```

Compute specificity of predictions at threshold.

Args:

    expected (np.array): ground truth, size (n_examples)
    preds (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    specificity (float): probability that the test outputs negative given that the case is actually negative


## get_ppv
```python
get_ppv(expected, preds, threshold=0.5)
```

Compute PPV of predictions at threshold.

Args:

    expected (np.array): ground truth, size (n_examples)
    preds (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    PPV (float): positive predictive value of predictions at threshold


## get_npv
```python
get_npv(expected, preds, threshold=0.5)
```

Compute NPV of predictions at threshold.

Args:

    expected (np.array): ground truth, size (n_examples)
    preds (np.array): model output, size (n_examples)
    threshold (float): cutoff value for positive prediction from model

Returns:

    NPV (float): negative predictive value of predictions at threshold


## compute_class_freqs
```python
compute_class_freqs(labels)
```

Compute positive and negative frequences for each class.

Args:

    labels (np.array): matrix of labels, size (num_examples, num_classes)

Returns:

    positive_frequencies (np.array): array of positive frequences for each
                                     class, size (num_classes)
    negative_frequencies (np.array): array of negative frequences for each
                                     class, size (num_classes)



## get_weighted_loss
```python
get_weighted_loss(pos_weights, neg_weights, epsilon=1e-07)
```

Return weighted loss function given negative weights and positive weights.

Args:

  pos_weights (np.array): array of positive weights for each class, size (num_classes)
  neg_weights (np.array): array of negative weights for each class, size (num_classes)

Returns:

  weighted_loss (function): weighted loss function


## generate_evaluation_report
```python
generate_evaluation_report(CLASS_NAMES,
                           predictions,
                           groundTruth=None,
                           generator=None,
                           returnPlot=True,
                           showPlot=True,
                           printStat=True,
                           **kwargs)
```

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

