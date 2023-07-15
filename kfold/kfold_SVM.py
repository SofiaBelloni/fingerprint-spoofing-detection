import sys
sys.path.append('C:\\Users\\ferla\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from SVM_classifier import *
from project import *
from metrics import *

def k_fold_cv(features_train, labels_train, k, mode, C, pT):
    #1. Dividere il training set in K fold
    dim = features_train.shape[1]
    max_elem = dim // k
    features_folds = []
    labels_folds = []

    SVM_znorm_scores = []
    SVM_scores = []
    labels = []
    for i in range(k):
        if i != k - 1:
            features_folds.append(features_train[:, i*max_elem : (i+1)*max_elem])
            labels_folds.append(labels_train[i*max_elem : (i+1)*max_elem])
        else:
            features_folds.append(features_train[:, i*max_elem::])
            labels_folds.append(labels_train[i*max_elem::])
    #Compute il dataset
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

        ### SVM ###
        svm = SupportVectorMachines(train, labels_training, C, pT)
        score= svm.compute_scores(eval, mode)
        SVM_scores.append(score)
        ### SVM with z-norm features ###
        svm_znorm = SupportVectorMachines(znorm_train, labels_training, C, pT)
        score= svm_znorm.compute_scores(znorm_eval, mode)
        SVM_znorm_scores.append(score)        
        labels.append(labels_eval)
        
    SVM_scores = np.hstack(SVM_scores)
    SVM_znorm_scores = np.hstack(SVM_znorm_scores)
    labels = np.hstack(labels)
    
    
    minDCF_SVM_score = compute_minDCF(pi, Cfn, Cfp, SVM_scores, labels)
    minDCF_SVM_znorm_score = compute_minDCF(pi, Cfn, Cfp, SVM_znorm_scores, labels)
    
    return minDCF_SVM_score, minDCF_SVM_znorm_score


features_train, labels_train = load_dataset('Train.txt')
features_train, labels_train = shuffle_dataset(features_train, labels_train)


#        SVM: linear, znorm, pca=7
pt = 0.7
#pt = 0.3
C = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10, 100]

#C = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1]
svm_C_results = []
znorm_SVM_C_results = []
svm_PCA_C_results = []
znorm_SVM_PCA_C_results = []
pca_dim = 7
P, _ = PCA(features_train, pca_dim)
y = np.dot(P.T, features_train)
for c in C:
    svm_pca, znorm_svm_pca = k_fold_cv(y, labels_train, 10, "linear", c, pt)
    svm, znorm_svm=k_fold_cv(features_train, labels_train, 10, "linear", c, pt)    
    svm_C_results.append(svm)
    znorm_SVM_C_results.append(znorm_svm)
    znorm_SVM_PCA_C_results.append(znorm_svm_pca)
    svm_PCA_C_results.append(svm_pca)
    print(f'SVM minDCF {svm} - C = {c}')
    print(f'ZNORM_SVM minDCF {znorm_svm} - C = {c}')
    print(f'SVM WITH PCA=7 minDCF {svm_pca} - C = {c}')
    print(f'ZNORM_SVM WITH PCA=7 minDCF {znorm_svm_pca} - C = {c}')

                    ### PLOT linear SVM ###
plt.figure(figsize=(10,6))
plt.plot(range(len(C)), svm_C_results, color='blue', label='linear SVM')
plt.plot(range(len(C)), svm_PCA_C_results, linestyle='-.', color='blue', label='linear SVM w/PCA=7')
plt.plot(range(len(C)), znorm_SVM_C_results, color='red', label='linear SVM (z-norm)')
plt.plot(range(len(C)), znorm_SVM_PCA_C_results, linestyle='-.', color='red', label='linear SVM w/PCA=7 (z-norm)')
plt.grid()
plt.xlabel('C')
plt.ylabel('minDCF')
plt.legend()
plt.xticks(range(len(C)), C)
plt.savefig('lSVM-vs-lSVM_norm-w-pca=7-pi=09-pt=07.png')
plt.show()


#        SVM: polynomial, znorm, pca=7
svm_C_results = []
znorm_SVM_C_results = []
svm_PCA_C_results = []
znorm_SVM_PCA_C_results = []
pca_dim = 7
P, _ = PCA(features_train, pca_dim)
y = np.dot(P.T, features_train)
for c in C:
    svm_pca, znorm_svm_pca = k_fold_cv(y, labels_train, 10, "polynomial", c, pt)
    svm, znorm_svm=k_fold_cv(features_train, labels_train, 10, "polynomial", c, pt)    
    svm_C_results.append(svm)
    znorm_SVM_C_results.append(znorm_svm)
    znorm_SVM_PCA_C_results.append(znorm_svm_pca)
    svm_PCA_C_results.append(svm_pca)
    print(f'SVM polynomial minDCF {svm} - C = {c}')
    print(f'ZNORM_SVM polynomial minDCF {znorm_svm} - C = {c}')
    print(f'SVM WITH PCA=7 polynomial minDCF {svm_pca} - C = {c}')
    print(f'ZNORM_SVM WITH polynomial PCA=7 minDCF {znorm_svm_pca} - C = {c}')

                    ### PLOT polynomial SVM ###
plt.figure(figsize=(10,6))
plt.plot(range(len(C)), svm_C_results, color='blue', label='polynomial SVM')
plt.plot(range(len(C)), svm_PCA_C_results, linestyle='-.', color='blue', label='polynomial SVM w/PCA=7')
plt.plot(range(len(C)), znorm_SVM_C_results, color='red', label='polynomial SVM (z-norm)')
plt.plot(range(len(C)), znorm_SVM_PCA_C_results, linestyle='-.', color='red', label='polynomial SVM w/PCA=7 (z-norm)')
plt.grid()
plt.xlabel('C')
plt.ylabel('minDCF')
plt.legend()
plt.xticks(range(len(C)), C)
plt.savefig('pSVM-vs-pSVM_norm-w-pca=7-pi=09-pt=07.png')
plt.show()


#        SVM: RBF, znorm, pca=7
svm_C_results = []
znorm_SVM_C_results = []
svm_PCA_C_results = []
znorm_SVM_PCA_C_results = []
pca_dim = 7
P, _ = PCA(features_train, pca_dim)
y = np.dot(P.T, features_train)
for c in C:
    svm_pca, znorm_svm_pca = k_fold_cv(y, labels_train, 10, "RBF", c, pt)
    svm, znorm_svm=k_fold_cv(features_train, labels_train, 10, "RBF", c, pt)    
    svm_C_results.append(svm)
    znorm_SVM_C_results.append(znorm_svm)
    znorm_SVM_PCA_C_results.append(znorm_svm_pca)
    svm_PCA_C_results.append(svm_pca)
    print(f'SVM RBF minDCF {svm} - C = {c}')
    print(f'ZNORM_SVM RBF minDCF {znorm_svm} - C = {c}')
    print(f'SVM WITH PCA=7 RBF minDCF {svm_pca} - C = {c}')
    print(f'ZNORM_SVM WITH RBF PCA=7 minDCF {znorm_svm_pca} - C = {c}')

                    ### PLOT SVM - RBF###
plt.figure(figsize=(10,6))
plt.plot(range(len(C)), svm_C_results, color='blue', label='RBF SVM')
plt.plot(range(len(C)), svm_PCA_C_results, linestyle='-.', color='blue', label='RBF SVM w/PCA=7')
plt.plot(range(len(C)), znorm_SVM_C_results, color='red', label='RBF SVM (z-norm)')
plt.plot(range(len(C)), znorm_SVM_PCA_C_results, linestyle='-.', color='red', label='RBF SVM w/PCA=7 (z-norm)')
plt.grid()
plt.xlabel('C')
plt.ylabel('minDCF')
plt.legend()
plt.xticks(range(len(C)), C)
plt.savefig('rbfSVM-vs-rbfSVM_norm-w-pca=7-pi=09-pt=07.png')
plt.show()

