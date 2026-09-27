import numpy as np
import csv


def calculate_conf_matrix(preds, labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for single_pred, single_label in zip(preds, labels):
        if single_pred == single_label and single_label == 1:
            tp = tp + 1
        elif single_pred == single_label and single_label == 0:
            tn = tn + 1
        elif single_pred != single_label and single_label == 1:
            fn = fn + 1
        elif single_pred != single_label and single_label == 0:
            fp = fp + 1
    return tp, tn, fp, fn


def calculate_accuracy(preds, labels):
    # This function should take predictions and labels made by the model
    # and return a single float depicting the
    # accuracy
    tp, tn, fp, fn = calculate_conf_matrix(preds, labels)
    acc = (tp + tn)/(tp + tn + fp + fn)
    return acc


def calculate_recall(preds, labels):
    # This function should take predictions and labels made by the model
    # and return a single float depicting the
    # recall
    # See: https://en.wikipedia.org/wiki/Precision_and_recall
    tp, tn, fp, fn = calculate_conf_matrix(preds, labels)
    recall = tp / (tp + fn)
    return recall


def calculate_precision(preds, labels):
    # This function should take predictions and labels made by the model
    # and return a single float depicting the
    # precision
    # See: https://en.wikipedia.org/wiki/Precision_and_recall
    tp, tn, fp, fn = calculate_conf_matrix(preds, labels)
    precision = tp / (tp + fp)
    return precision


def calculate_f1_score(preds, labels):
    # This function should take predictions and labels made by the model
    # and return a single float depicting the
    # F1-score
    # See: https://en.wikipedia.org/wiki/F-score
    tp, tn, fp, fn = calculate_conf_matrix(preds, labels)
    f1_score = 2*tp / (2*tp + fp + fn)
    return f1_score


def save_results(preds, labels):
    # This function should save the performance results of the model to a csv
    # file. At the very least, it should save accuracy and f1_score
    # for both train and val datasets. You are flexible in how you handle the
    # saving process. You might not even use this function, feel free to save
    # however you want to save results.
    accuracy = calculate_accuracy(preds, labels)
    recall = calculate_recall(preds, labels)
    precision = calculate_precision(preds, labels)
    f1_score = calculate_f1_score(preds, labels)

    header = ['Accuracy', 'Recall', 'Precision', 'F1 score']
    file = open('performance results.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerow([accuracy, recall, precision, f1_score])
    file.close()
