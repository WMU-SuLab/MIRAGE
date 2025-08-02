import pandas as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    matthews_corrcoef, roc_curve, auc

def count_metrics_binary_classification(y_true: list, y_pred: list, y_score: list):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("matrix",confusion_matrix(y_true, y_pred))
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    ks = abs(tpr - fpr)
    sp = 1 - fpr
    SP = tn / (tn + fp)
    acc = (tp + tn) / (tp + fn + tn + fp)
    ACC = accuracy_score(y_true, y_pred)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    MCC = matthews_corrcoef(y_true, y_pred)
    precision = tp / (tp + fp)
    Precision = precision_score(y_true, y_pred)
    recall = tp / (tp + fn)
    Recall = recall_score(y_true, y_pred)
    f1 = 2 * precision * recall / (precision + recall)
    F1 = f1_score(y_true, y_pred)
    FPR, TPR, thresholds = roc_curve(y_true, y_score)
    KS = abs(FPR - TPR).max()
    AUC = auc(FPR, TPR)
    auc_score = roc_auc_score(y_true, y_score)
    return locals()


def count_metrics_multi_classification(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # auc_score = auc(fpr, tpr)
    auc_score = roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
    return acc, precision, recall, f1, auc_score
