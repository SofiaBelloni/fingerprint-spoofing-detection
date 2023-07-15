import sys
sys.path.append('c:\\Users\\ferla\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from logistic_regression import *
from project import *
from metrics import *



def k_fold_cv(features_train, labels_train, k, l, quadratic = False, pt=None):
    #1. Dividere il training set in K fold
    dim = features_train.shape[1]
    max_elem = dim // k
    features_folds = []
    labels_folds = []

    LOGREG_scores = []
    ZNORM_LOGREG_scores = []
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
        if (quadratic):
            train, labels_training = quadratic_features_expansion(train, labels_training)
            eval, labels_eval = quadratic_features_expansion(eval, labels_eval)
        znorm_train, znorm_eval = znorm(train, eval)
        logreg = logRegClass(train, labels_training, l, pt)
        znorm_logreg = logRegClass(znorm_train, labels_training, l, pt)
        ZNORM_LOGREG_scores.append(znorm_logreg.compute_scores(znorm_eval))
        LOGREG_scores.append(logreg.compute_scores(eval))
        labels.append(labels_eval)
        
    LOGREG_scores = np.hstack(LOGREG_scores)
    ZNORM_LOGREG_scores = np.hstack(ZNORM_LOGREG_scores)
    labels = np.hstack(labels)
    
    minDCF_LOGREG_score = compute_minDCF(pi, Cfn, Cfp, LOGREG_scores, labels)
    minDCF_ZNORM_LOGREG_score = compute_minDCF(pi, Cfn, Cfp, ZNORM_LOGREG_scores, labels)

    return minDCF_LOGREG_score, minDCF_ZNORM_LOGREG_score


features_train, labels_train = load_dataset('Train.txt')
features_train, labels_train = shuffle_dataset(features_train, labels_train)


#                                                    Logistic Regression: linear, znorm
# lam = [1E-1]
# logreg_lambda_results = []
# znorm_logreg_lambda_results = []
# for l in lam:
#     logreg, znorm_logreg=k_fold_cv(features_train, labels_train, 10, l)
#     logreg_lambda_results.append(logreg)
#     znorm_logreg_lambda_results.append(znorm_logreg)
#     print(f'LOGREG minDCF {logreg} - LAMBDA = {l}')
#     print(f'ZNORM_LOGREG minDCF {znorm_logreg} - LAMBDA = {l}')

# plt.figure(figsize=(10,6))
# plt.plot(range(len(lam)), logreg_lambda_results, color='blue', label='LogReg')
# plt.plot(range(len(lam)), znorm_logreg_lambda_results, color='red', label='LogReg (z-norm)')
# plt.grid()
# plt.xlabel('λ')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(lam)), lam)
# plt.savefig('logreg-vs-logregz_norm.png')
# plt.show()




                                                    # Logistic Regression: linear, znorm, pca=6,7,8 (da rimuovere)
# lam = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10]
# pca_dim = [6,7,8]
# plt.figure(figsize=(10,6))
# for p in pca_dim:
#     P, _ = PCA(features_train, p)
#     y = np.dot(P.T, features_train)
#     logreg_lambda_results = []
#     znorm_logreg_lambda_results = []
#     for l in lam:
#         logreg, znorm_logreg=k_fold_cv(y, labels_train, 10, l)
#         logreg_lambda_results.append(logreg)
#         znorm_logreg_lambda_results.append(znorm_logreg)
#         print(f'LOGREG minDCF {logreg} - LAMBDA = {l}')
#         print(f'ZNORM_LOGREG minDCF {znorm_logreg} - LAMBDA = {l}')
#     plt.plot(range(len(lam)), logreg_lambda_results, color='blue', label=f'LogReg - PCA = {p}')
#     plt.plot(range(len(lam)), znorm_logreg_lambda_results, color='red', label=f'LogReg (z-norm) - PCA = {p}')
# plt.grid()
# plt.xlabel('λ')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(lam)), lam)
# plt.savefig('logreg-vs-logregz_norm-some_pca.png')
# plt.show()


#                                                    Logistic Regression: linear, znorm, pca=8
# lam = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10]
# logreg_lambda_results = []
# znorm_logreg_lambda_results = []
# logreg_pca_lambda_results = []
# znorm_logreg_pca_lambda_results = []
# pca_dim = 7
# P, _ = PCA(features_train, pca_dim)
# y = np.dot(P.T, features_train)
# for l in lam:
#     logreg_pca, znorm_logreg_pca = k_fold_cv(y, labels_train, 10, l)
#     logreg, znorm_logreg=k_fold_cv(features_train, labels_train, 10, l)
#     logreg_lambda_results.append(logreg)
#     logreg_pca_lambda_results.append(logreg_pca)
#     znorm_logreg_pca_lambda_results.append(znorm_logreg_pca)
#     znorm_logreg_lambda_results.append(znorm_logreg)
#     print(f'LOGREG minDCF {logreg} - LAMBDA = {l}')
#     print(f'ZNORM_LOGREG minDCF {znorm_logreg} - LAMBDA = {l}')
#     print(f'LOGREG WITH PCA=6 minDCF {logreg_pca} - LAMBDA = {l}')
#     print(f'ZNORM_LOGREG WITH PCA=6 minDCF {znorm_logreg_pca} - LAMBDA = {l}')

#                     ### PLOT LOGREG VS LOGREG ZNORM ###
# plt.figure(figsize=(10,6))
# plt.plot(range(len(lam)), logreg_lambda_results, color='blue', label='LogReg')
# plt.plot(range(len(lam)), logreg_pca_lambda_results, linestyle='-.', color='blue', label='LogReg w/PCA=7')
# plt.plot(range(len(lam)), znorm_logreg_lambda_results, color='red', label='LogReg (z-norm)')
# plt.plot(range(len(lam)), znorm_logreg_pca_lambda_results, linestyle='-.', color='red', label='LogReg w/PCA=7 (z-norm)')
# plt.grid()
# plt.xlabel('λ')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(lam)), lam)
# plt.savefig('logreg-vs-logregz_norm-w-pca=7-pi=09.png')
# plt.show()


# #                                                     # Logistic Regression: quadratic, znorm
# lam = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10]
# logreg_lambda_results = []
# znorm_logreg_lambda_results = []
# for l in lam:
#     logreg, znorm_logreg=k_fold_cv(features_train, labels_train, 10, l, True)
#     logreg_lambda_results.append(logreg)
#     znorm_logreg_lambda_results.append(znorm_logreg)
#     print(f'Q-LOGREG minDCF {logreg} - LAMBDA = {l}')
#     print(f'Q-ZNORM_LOGREG minDCF {znorm_logreg} - LAMBDA = {l}')

# plt.figure(figsize=(10,6))
# plt.plot(range(len(lam)), logreg_lambda_results, color='blue', label='Q-LogReg')
# plt.plot(range(len(lam)), znorm_logreg_lambda_results, color='red', label='Q-LogReg (z-norm)')
# plt.grid()
# plt.xlabel('λ')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(lam)), lam)
# plt.savefig('qlogreg-vs-qlogregz_norm-pi=09.png')
# plt.show()

                                # Linear, znorm, prior weighted, pca=7
# lam = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10]
# logreg_lambda_results = []
# znorm_logreg_lambda_results = []
# logreg_pca_lambda_results = []
# znorm_logreg_pca_lambda_results = []
# pca_dim = 7
# P, _ = PCA(features_train, pca_dim)
# y = np.dot(P.T, features_train)
# for l in lam:
#     logreg_pca, znorm_logreg_pca = k_fold_cv(y, labels_train, 10, l, pt=pi)
#     logreg, znorm_logreg=k_fold_cv(features_train, labels_train, 10, l, pt=pi)
#     logreg_lambda_results.append(logreg)
#     logreg_pca_lambda_results.append(logreg_pca)
#     znorm_logreg_pca_lambda_results.append(znorm_logreg_pca)
#     znorm_logreg_lambda_results.append(znorm_logreg)
#     print(f'LOGREG minDCF {logreg} - LAMBDA = {l}')
#     print(f'ZNORM_LOGREG minDCF {znorm_logreg} - LAMBDA = {l}')
#     print(f'LOGREG WITH PCA=6 minDCF {logreg_pca} - LAMBDA = {l}')
#     print(f'ZNORM_LOGREG WITH PCA=6 minDCF {znorm_logreg_pca} - LAMBDA = {l}')

#                     ### PLOT LOGREG VS LOGREG ZNORM ###
# plt.figure(figsize=(10,6))
# plt.plot(range(len(lam)), logreg_lambda_results, color='blue', label='PW-LogReg')
# plt.plot(range(len(lam)), logreg_pca_lambda_results, linestyle='-.', color='blue', label='PW-LogReg w/PCA=7')
# plt.plot(range(len(lam)), znorm_logreg_lambda_results, color='red', label='PW-LogReg (z-norm)')
# plt.plot(range(len(lam)), znorm_logreg_pca_lambda_results, linestyle='-.', color='red', label='PW-LogReg w/PCA=7 (z-norm)')
# plt.grid()
# plt.xlabel('λ')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(lam)), lam)
# plt.savefig('PWlogreg-vs-PWlogregz_norm-w-pca=7-pi=09.png')
# plt.show()

#                         # linear, znorm, prior-weighted
lam = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10]
logreg_lambda_results = []
znorm_logreg_lambda_results = []
for l in lam:
    logreg, znorm_logreg=k_fold_cv(features_train, labels_train, 10, l, pt=0.7)
    logreg_lambda_results.append(logreg)
    znorm_logreg_lambda_results.append(znorm_logreg)
    print(f'LOGREG minDCF {logreg} - LAMBDA = {l}')
    print(f'ZNORM_LOGREG minDCF {znorm_logreg} - LAMBDA = {l}')

plt.figure(figsize=(10,6))
plt.plot(range(len(lam)), logreg_lambda_results, color='blue', label='LogReg')
plt.plot(range(len(lam)), znorm_logreg_lambda_results, color='red', label='LogReg (z-norm)')
plt.grid()
plt.xlabel('λ')
plt.ylabel('minDCF')
plt.legend()
plt.xticks(range(len(lam)), lam)
plt.savefig('PWlogreg-vs-PWlogregz_norm-pi=05-pt=07.png')
plt.show()













