from project import *
from MVG_classifier import MVG
from logistic_regression import logRegClass, quadratic_features_expansion
from metrics import *
import random


pi = 0.5
Cfn = 1
Cfp = 10

def k_fold_cv(features_train, labels_train, k):
    #1. Dividere il training set in K fold
    dim = features_train.shape[1]
    max_elem = dim // k
    features_folds = []
    labels_folds = []

    MVG_scores = []
    MVG_NB_scores = []
    MVG_TIED_scores = []
    LOG_REG_scores = []
    LOG_REG_EXPANDED_scores = []
    SVM_scores = []
    GMM_scores = []

    labels = []
    labels_test_expanded = []
    for i in range(k):
        if i != k - 1:
            features_folds.append(features_train[:, i*max_elem : (i+1)*max_elem])
            labels_folds.append(labels_train[i*max_elem : (i+1)*max_elem])
        else:
            features_folds.append(features_train[:, i*max_elem::])
            labels_folds.append(labels_train[i*max_elem::])
    #Calcolo il dataset
    eval = []
    labels_eval = []
    for i in range(k):
        train = []
        labels_training = []
        for j in range(k):
            if (i == j):
                eval = features_folds[i] 
                labels_eval = labels_folds[i]
            else:
                train.append(features_folds[j])
                labels_training.append(labels_folds[j])
        train = np.hstack(train)
        labels_training = np.hstack(labels_training)
        # Qui noi alleniamo
        # Chiamare i vari metodi di classificazione
        ### MVG ###
        mvg = MVG(train, labels_training)
        MVG_scores.append(mvg.llr(eval, 'standard'))
        MVG_NB_scores.append(mvg.llr(eval, 'nb'))
        MVG_TIED_scores.append(mvg.llr(eval, 'tied'))
        labels.append(labels_eval)
        
        ### LOGISTIC REGRESSION ####
        lr = logRegClass(train, labels_training, 0.01)
        LOG_REG_scores.append(lr.compute_scores(eval))

        features_expanded, labels_expanded = quadratic_features_expansion(train, labels_training)
        ftest_expanded, ltest_expanded = quadratic_features_expansion(eval, labels_eval)
        lr = logRegClass(features_expanded, labels_expanded, 0.01)
        LOG_REG_EXPANDED_scores.append(lr.compute_scores(ftest_expanded))
        labels_test_expanded.append(ltest_expanded)

        ### SVM ###

        ### GMM ###
    MVG_scores = np.hstack(MVG_scores)
    MVG_NB_scores = np.hstack(MVG_NB_scores)
    MVG_TIED_scores = np.hstack(MVG_TIED_scores)
    LOG_REG_scores = np.hstack(LOG_REG_scores)
    LOG_REG_EXPANDED_scores = np.hstack(LOG_REG_EXPANDED_scores)
    labels = np.hstack(labels)
    labels_test_expanded = np.hstack(labels_test_expanded)
    
    minDCF_MVG_score = compute_minDCF(pi, Cfn, Cfp, MVG_scores, labels)
    minDCF_MVG_NB_score = compute_minDCF(pi, Cfn, Cfp, MVG_NB_scores, labels)
    minDCF_MVG_TIED_score = compute_minDCF(pi, Cfn, Cfp, MVG_TIED_scores, labels)
    minDCF_LOGREG_score = compute_minDCF(pi, Cfn, Cfp, LOG_REG_scores, labels)
    minDCF_LOGREG_FEATURE_EXPANSION_score = compute_minDCF(pi, Cfn, Cfp, LOG_REG_EXPANDED_scores, labels_test_expanded)

    #minDCF_SVM_TIED_score = compute_minDCF(pi, Cfn, Cfp, SVM_scores, labels)
    #minDCF_GMM_TIED_score = compute_minDCF(pi, Cfn, Cfp, GMM_scores, labels)
    return minDCF_MVG_score, minDCF_MVG_NB_score, minDCF_MVG_TIED_score, minDCF_LOGREG_score, minDCF_LOGREG_FEATURE_EXPANSION_score


    
features_train, labels_train = load_dataset('Train.txt')
features_train, labels_train = shuffle_dataset(features_train, labels_train)
features_test, labels_test = load_dataset('Test.txt')
MVG_STD, mvg_nb, mvg_tied, logreg, log_exp= k_fold_cv(features_train, labels_train, 10)

print(f'MVG minDCF accuracy {MVG_STD}')
print(f'MVG_NB minDCF accuracy {mvg_nb}')
print(f'MVG MVG_TIED accuracy {mvg_tied}')
print(f'LOGREG accuracy {logreg}')
print(f'LOGREG accuracy {log_exp}')

