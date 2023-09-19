import sys
sys.path.append('c:\\Users\\ferla\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from logistic_regression import *
from project import *
from metrics import *
from GMM_classifier import GMM_classifier
from MVG_classifier import *
from SVM_classifier import *

def k_fold_cv(features_train, labels_train, k, n=2):
    #1. Dividere il training set in K fold
    dim = features_train.shape[1]
    max_elem = dim // k
    features_folds = []
    labels_folds = []

    znorm_GMM_naive_tied_scores = []
    znorm_MVG_standard_scores = []
    znorm_SVM_linear_scores = []

    labels = []
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
        znorm_train, znorm_eval = znorm(train, eval)
        znorm_GMM_model = GMM_classifier(znorm_train, labels_training, n)
        znorm_MVG_model = MVG(znorm_train, labels_training)
        znorm_SVM_model = SupportVectorMachines(znorm_train, labels_training, 0.1, 0.5)
        znorm_score_naive_tied = znorm_GMM_model.compute_scores(znorm_eval, 'naive', 'tied')    
        znorm_GMM_naive_tied_scores.append(znorm_score_naive_tied)
        znorm_SVM_linear_score = znorm_SVM_model.compute_scores(znorm_eval, 'linear')
        labels.append(labels_eval)
        znorm_MVG_standard_scores.append(znorm_MVG_model.llr(znorm_eval, 'standard'))
        znorm_SVM_linear_scores.append(znorm_SVM_linear_score)

    znorm_GMM_naive_tied_scores = np.hstack(znorm_GMM_naive_tied_scores)
    znorm_MVG_standard_scores = np.hstack(znorm_MVG_standard_scores)
    znorm_SVM_linear_scores = np.hstack(znorm_SVM_linear_scores)
    labels = np.hstack(labels)
    return znorm_MVG_standard_scores, znorm_GMM_naive_tied_scores, znorm_SVM_linear_scores, labels


features_train, labels_train = load_dataset('Train.txt')
features_train, labels_train = shuffle_dataset(features_train, labels_train)
P, _ = PCA(features_train, 7)
y = np.dot(P.T, features_train)
gmmScores, mvgScores, svmScores, labels_train = k_fold_cv(y, labels_train, 10)
bayes_error_plot(gmmScores, labels_train, 'gmm')
bayes_error_plot(mvgScores, labels_train, 'mvg')
bayes_error_plot(svmScores, labels_train, 'svm')

# Calibration
svmScores, labels_train = shuffle_dataset(svmScores, labels_train)
caliTrain = svmScores[:, :int(svmScores.shape[1] * 0.75)]
caliTest = svmScores[:, int(svmScores.shape[1] * 0.75):]
caliLabelTrain = labels_train[:int(len(labels_train) * 0.75)]
caliLabelTest = labels_train[int(len(labels_train) * 0.75):]


laambda = 0.01
caliLogReg = logRegClass(caliTrain, caliLabelTrain, laambda, 0.5)

w, b, calibration= caliLogReg.compute_calibration_score(caliTrain, caliLabelTrain, caliTest)
caliLogReg_scores = np.dot(w.T, svmScores) + b - calibration

bayes_error_plot(caliLogReg_scores, labels_train, 'svm-calibrated')


