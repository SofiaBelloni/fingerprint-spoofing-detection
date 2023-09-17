import numpy as np
import matplotlib.pyplot as plt


def binary_confusion_matrix(true_labels, predicted_labels):
    # Calcolo delle variabili necessarie
    C = np.zeros((2, 2))
    C[1,1] = int(sum([1 for true, predicted in zip(true_labels, predicted_labels) if true == 1 and predicted == 1]))
    C[0,0] = int(sum([1 for true, predicted in zip(true_labels, predicted_labels) if true == 0 and predicted == 0]))
    C[1,0] = int(sum([1 for true, predicted in zip(true_labels, predicted_labels) if true == 0 and predicted == 1]))
    C[0,1] = int(sum([1 for true, predicted in zip(true_labels, predicted_labels) if true == 1 and predicted == 0]))
    return C


def compute_threshold(pi, Cfn, Cfp):
    return -1*np.log((pi*Cfn)/((1-pi)*Cfp))

def compute_FNR(CM):
    return CM[0,1] / (CM[0,1] + CM[1,1])

def compute_FPR(CM):
    return CM[1,0] / (CM[1,0] + CM[0,0])

def binary_emp_Bayes_risk(pi, Cfn, Cfp, CM):
    # un-normalized DCF
    #FNR = false rejection / miss rate = FN / (FN+TP)
    FNR = compute_FNR(CM)
    #FPR = false acceptance = FP / (FP*TN)
    FPR = compute_FPR(CM)
    return pi*Cfn*FNR + (1-pi)*Cfp*FPR

def optimal_system_risk(pi, Cfn, Cfp):
    return min(pi*Cfn, (1-pi)*Cfp)

def binary_normalized_Bayes_risk(pi, Cfn, Cfp, CM):
    return binary_emp_Bayes_risk(pi, Cfn, Cfp, CM) / optimal_system_risk(pi, Cfn, Cfp)

def compute_minDCF(pi, Cfn, Cfp, scores, labels):
     # we can compute the optimal threshold for a given application on the same validation set, and we 
    # use such threshold for the test population
    thresholds = np.array(scores)
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    DCFs = []
    for t in thresholds:
        pred = np.int32(scores > t)
        CM = binary_confusion_matrix(labels, pred)
        DCFs.append(binary_normalized_Bayes_risk(pi, Cfn, Cfp, CM))
    return np.array(DCFs).min()

def compute_actDCF(pi, Cfn, Cfp, scores, labels):
    threshold = -np.log((pi*Cfn)/((1-pi)*Cfp))
    pred = np.int32(scores > threshold)
    CM = binary_confusion_matrix(labels, pred)
    return binary_normalized_Bayes_risk(pi, Cfn, Cfp, CM)

def bayes_error_plot(scores, labels, title):
    p = np.linspace(-3,3,21)
    Cfn = 1
    Cfp = 1
    minDCF = []
    actDCF = []
    for pi in p:
        pi_tilde = 1 / (1 + np.exp(-pi))
        minDCF.append(compute_minDCF(pi_tilde, Cfn, Cfp, scores, labels))
        actDCF.append(compute_actDCF(pi_tilde, Cfn, Cfp, scores, labels))
    plt.plot(p, minDCF, label='minDCF', color='b')
    plt.plot(p, actDCF, label='actDCF', color='r')
    plt.legend()
    plt.grid()
    plt.savefig(f'{title}.png')
    plt.show()
   