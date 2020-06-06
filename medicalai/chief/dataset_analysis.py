import pandas as pd
import numpy as np
def check_dataset_leakage(trainDF, testDF, columnName):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        trainDF: (dataframe): dataframe describing first dataset
        testDF: (dataframe): dataframe describing second dataset
        columnName: (str): string name of column to check for data leakage
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
    if isinstance(trainDF,str):
        df1 = pd.read_csv(trainDF)
    else:
        df1 = trainDF
    
    if isinstance(testDF,str):
        df2 = pd.read_csv(testDF)
    else:
        df2 = testDF

    df1_unique = set(df1[columnName].values)
    df2_unique = set(df2[columnName].values)
    
    duplicates = list(df1_unique.intersection(df2_unique))

    if len(duplicates)>0:
        leakage = True
        print('[WARNING]: Data Leakage found between Train and Test Dataset')
    else:
        leakage = False
        print('[INFO]: NO Data Leakage found between Train and Test Dataset')
    
    return leakage

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
