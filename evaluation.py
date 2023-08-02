import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, \
    recall_score, f1_score, precision_score, roc_curve, classification_report, confusion_matrix, auc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_metrics(y_true, y_pred):

    # get sensitivity, specificity, and accuracy
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            TN = TN + 1
        elif true == 0 and pred == 1:
            FP = FP + 1
        elif true == 1 and pred == 1:
            TP = TP + 1
        elif true == 1 and pred == 0:
            FN = FN + 1

    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1, specificity

def get_cm(y_true, y_pred):

    # plt.style.use('classic')
    plt.figure()
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(confusion)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    print(indices)

    plt.tick_params(labelsize=16)
    plt.xticks(indices, ['good', 'poor'])
    plt.yticks(indices, ['good', 'poor'])

    plt.colorbar()

    plt.xlabel('predicted', fontdict={"family": "Arial", "size": 16})
    plt.ylabel('ground truth', fontdict={"family": "Arial", "size": 16})
    plt.title('confusion matrix', fontdict={"family": "Arial", "size": 16})

    for first_index in range(len(confusion)):  
        for second_index in range(len(confusion[first_index])):  
            print(first_index, second_index)
            # plt.text(first_index, second_index, confusion[first_index][second_index])
            plt.text(first_index, second_index, confusion[second_index][first_index], size=16)

    # plt.show()
    plt.savefig('results/{}_confusion_matrix.png'.format(aspect))
    plt.close('all')

def get_cm_guidance(y_true, y_pred):

    plt.figure()
    confusion = confusion_matrix(y_true, y_pred, labels=['finish', 'recapture', 'referral'])
    print(confusion)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    print(indices)


    plt.tick_params(labelsize=16)
    plt.xticks(indices, ['finish', 'recapture', 'referral'])
    plt.yticks(indices, ['finish', 'recapture', 'referral'])

    plt.colorbar()

    plt.xlabel('predicted', fontdict={"family": "Arial", "size": 16})
    plt.ylabel('ground truth', fontdict={"family": "Arial", "size": 16})
    plt.title('confusion matrix', fontdict={"family": "Arial", "size": 16})


    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            print(first_index, second_index)
            plt.text(first_index, second_index, confusion[second_index][first_index], size=16)

    plt.tight_layout()
    plt.savefig('results/guidance_confusion_matrix.png')
    plt.close('all')

def plot_ROC(y_true, y_pred_prob, aspect):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)

    plt.style.use('seaborn')
    plt.figure()
    plt.plot(fpr, tpr, label='Area Under Curve (AUC) = {0:0.2f}'.format(auc_score), color='red')
    plt.tick_params(labelsize=14)  ## control ticks' font size
    plt.xlim([0.0, 1.0])
    plt.ylim([0.1, 1.0])
    tick_label_font = {
        # 'family':'Times New Roman',
        'style': 'normal',
        'weight': 'normal',
        'color': 'k',
        'size': 14}
    plt.xlabel('False Positive Rate', tick_label_font)
    plt.ylabel('True Positive Rate', tick_label_font)
    plt.title('ROC', fontdict={'fontsize': 18})
    plt.legend(loc="lower right", prop={'size': 14})
    plt.savefig(r'results/{}_ROC.png'.format(aspect), dpi=300)


parser = argparse.ArgumentParser(description = 'configuration')
parser.add_argument('--mode',  type=int, default=1, choices=[1, 2], help='1 indicates quality classification and 2 indicates real-time guidance')
args = parser.parse_args()

if args.mode==1:
    fn_list = []
    gt_list = []
    all_pred = []
    all_pred_proba = []
    df_gt = pd.read_excel(r'results/quality_gt.xlsx', header=0, dtype='str', engine='openpyxl', index_col=0)
    df_pred = pd.read_excel(r'results/quality_pred.xlsx', header=0, dtype='str', engine='openpyxl', index_col=0)
    df_proba = pd.read_excel(r'results/quality_proba.xlsx', header=0, dtype='str', engine='openpyxl', index_col=0)

    df_gt = df_gt.sort_index(ascending=False)
    df_pred = df_pred.sort_index(ascending=False)
    df_proba = df_proba.sort_index(ascending=False)

    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_specificity = []
    for aspect in df_pred.columns:
        y_true = df_gt[aspect].astype(int).values.tolist()
        y_pred = df_pred[aspect].astype(int).values.tolist()
        get_cm(y_true, y_pred)  # generate confusion matrix

    for aspect in df_pred.columns:
        y_true = df_gt[aspect].astype(int).values.tolist()
        y_pred = df_pred[aspect].astype(int).values.tolist()
        y_pred_prob = df_proba[aspect].astype(float).values.tolist()
        plot_ROC(y_true, y_pred_prob, aspect) # plot and save ROC
        accuracy, precision, recall, f1, specificity = get_metrics(y_true, y_pred) # obtain metrics

        all_accuracy.append(accuracy)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_specificity.append(specificity)

    df = pd.DataFrame({'accuracy': all_accuracy, 'precision': all_precision, 'recall(sensitivity)': all_recall,
                       'f1': all_f1,  'specificity': all_specificity}, index=df_pred.columns)
    df.to_excel('results/quality_metrics.xlsx')

elif args.mode==2:
    class_to_indices = {'finish': 0, 'recapture':1, 'referral':2}
    fn_list = []
    gt_list = []
    all_pred = []
    df_gt = pd.read_excel(r'results/advice_gt.xlsx', header=0, dtype='str', engine='openpyxl', index_col=0)
    df_pred = pd.read_excel(r'results/advice_pred.xlsx', header=0, dtype='str', engine='openpyxl', index_col=0)
    df_gt = df_gt.sort_index(ascending=False)
    df_pred = df_pred.sort_index(ascending=False)

    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_specificity = []

    y_true = df_gt['guidance'].values.tolist()
    y_pred = df_pred['prediction'].values.tolist()
    get_cm_guidance(y_true, y_pred)

    for c in class_to_indices.keys():

        y_true_temp = [1 if i==c else 0 for i in y_true]
        y_pred_temp = [1 if i==c else 0 for i in y_pred]
        accuracy, precision, recall, f1, specificity = get_metrics(y_true_temp, y_pred_temp)
        all_accuracy.append(accuracy)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_specificity.append(specificity)

    df = pd.DataFrame({'accuracy': all_accuracy, 'precision': all_precision, 'recall(sensitivity)': all_recall,
                       'f1': all_f1, 'specificity': all_specificity}, index=class_to_indices.keys())
    df.to_excel('results/guidance_metrics.xlsx')
