from sklearn import metrics
import numpy as np

def accuracy(y_pred, y_true):
    """
    Calculate accuracy for multi-class classification.
    Args:
        y_pred (numpy.array): Predicted class indices or probability distribution
        y_true (numpy.array): True class indices
    """
    # For multi-class, y_pred contains logits/probabilities for each class
    # We need to get the predicted class with highest probability
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred.round()
    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred_classes)

def precision(y_pred, y_true):
    """
    Calculate precision for multi-class classification using macro averaging.
    Args:
        y_pred (numpy.array): Predicted class indices or probability distribution
        y_true (numpy.array): True class indices
    """
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred.round()
    return metrics.precision_score(y_true=y_true, y_pred=y_pred_classes, average='macro', zero_division=0)

def recall(y_pred, y_true):
    """
    Calculate recall for multi-class classification using macro averaging.
    Args:
        y_pred (numpy.array): Predicted class indices or probability distribution
        y_true (numpy.array): True class indices
    """
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred.round()
    return metrics.recall_score(y_true=y_true, y_pred=y_pred_classes, average='macro', zero_division=0)

def f1_score(y_pred, y_true):
    """
    Calculate F1 score for multi-class classification using macro averaging.
    Args:
        y_pred (numpy.array): Predicted class indices or probability distribution
        y_true (numpy.array): True class indices
    """
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred.round()
    return metrics.f1_score(y_true=y_true, y_pred=y_pred_classes, average='macro', zero_division=0)

def top_k_accuracy(y_pred, y_true, k=3):
    """
    Calculate top-k accuracy for multi-class classification.
    Args:
        y_pred (numpy.array): Probability distribution over classes
        y_true (numpy.array): True class indices
        k (int): Number of top predictions to consider
    """
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        return accuracy(y_pred, y_true)
        
    batch_size = len(y_true)
    topk_preds = np.argsort(-y_pred, axis=1)[:, :k]
    correct = 0
    for i in range(batch_size):
        if y_true[i] in topk_preds[i]:
            correct += 1
    return correct / batch_size

# The following functions are not directly applicable to multi-class problems
# but are kept for backward compatibility or potential one-vs-rest usage

def roc_auc(y_pred, y_true):
    """
    For multi-class, use one-vs-rest ROC AUC score.
    """
    # Check if this is multi-class with probabilities
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 2:
        try:
            return metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='macro', multi_class='ovr')
        except:
            # Fallback to weighted average if needed
            return metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='weighted', multi_class='ovr')
    # Binary case
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)

def pr_auc(y_pred, y_true):
    """
    For multi-class, average precision is calculated for each class and then averaged.
    """
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 2:
        # For multi-class, need to convert to one-hot encoding
        y_true_onehot = np.zeros((len(y_true), y_pred.shape[1]))
        for i, label in enumerate(y_true):
            y_true_onehot[i, int(label)] = 1
        return metrics.average_precision_score(y_score=y_pred, y_true=y_true_onehot, average='macro')
    return metrics.average_precision_score(y_score=y_pred, y_true=y_true)

# Curve functions may need modification for specific multi-class use cases
def precision_recall_curve(y_pred, y_true):
    # Only works for binary classification; kept for backwards compatibility
    precision, recall, thresholds = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    prc_dict = {'precision': precision, 'recall': recall, 'thresholds': thresholds}
    return prc_dict

def roc_curve(y_pred, y_true):
    # Only works for binary classification; kept for backwards compatibility
    fpr, tpr, thresholds = metrics.roc_curve(y_score=y_pred, y_true=y_true, pos_label=1)
    roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    return roc_dict