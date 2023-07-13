import sys
sys.path.append('c:\\Users\\ferla\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from MVG_classifier import *
from project import *
from metrics import *



def k_fold_cv(features_train, labels_train, k):
    #1. Dividere il training set in K fold
    dim = features_train.shape[1]
    max_elem = dim // k
    features_folds = []
    labels_folds = []

    MVG_scores = []
    MVG_NB_scores = []
    MVG_TIED_scores = []
    
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

        ### MVG ###
        mvg = MVG(train, labels_training)
        MVG_scores.append(mvg.llr(eval, 'standard'))
        MVG_NB_scores.append(mvg.llr(eval, 'nb'))
        MVG_TIED_scores.append(mvg.llr(eval, 'tied'))
        labels.append(labels_eval)
        
    MVG_scores = np.hstack(MVG_scores)
    MVG_NB_scores = np.hstack(MVG_NB_scores)
    MVG_TIED_scores = np.hstack(MVG_TIED_scores)
    labels = np.hstack(labels)
    
    minDCF_MVG_score = compute_minDCF(pi, Cfn, Cfp, MVG_scores, labels)
    minDCF_MVG_NB_score = compute_minDCF(pi, Cfn, Cfp, MVG_NB_scores, labels)
    minDCF_MVG_TIED_score = compute_minDCF(pi, Cfn, Cfp, MVG_TIED_scores, labels)

    return minDCF_MVG_score, minDCF_MVG_NB_score, minDCF_MVG_TIED_score



features_train, labels_train = load_dataset('Train.txt')
features_train, labels_train = shuffle_dataset(features_train, labels_train)

pca_dimensions = [5,6,7]
for pca in pca_dimensions:
    P, _ = PCA(features_train, pca)
    y = np.dot(P.T, features_train)
    mvg_std, mvg_nb, mvg_tied= k_fold_cv(y, labels_train, 10)
    print(f'MVG minDCF accuracy {mvg_std} - m = {pca}')
    print(f'MVG_NB minDCF accuracy {mvg_nb} - m = {pca}')
    print(f'MVG MVG_TIED accuracy {mvg_tied} - m = {pca}')
    print('\n')

mvg_std, mvg_nb, mvg_tied= k_fold_cv(features_train, labels_train, 10)
print(f'MVG minDCF accuracy {mvg_std} - NO PCA')
print(f'MVG_NB minDCF accuracy {mvg_nb} - NO PCA')
print(f'MVG MVG_TIED accuracy {mvg_tied} - NO PCA')





