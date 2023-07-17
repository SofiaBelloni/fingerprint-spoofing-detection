import sys
sys.path.append('C:\\Users\\ferla\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from SVM_classifier import *
from project import *
from metrics import *

def k_fold_cv(features_train, labels_train, k, mode, C, pT,  gamma=1):
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
        svm = SupportVectorMachines(train, labels_training, C, pT, gamma=gamma)
        score= svm.compute_scores(eval, mode)
        SVM_scores.append(score)
        ### SVM with z-norm features ###
        svm_znorm = SupportVectorMachines(znorm_train, labels_training, C, pT, gamma=gamma)
        score= svm_znorm.compute_scores(znorm_eval, mode)
        SVM_znorm_scores.append(score)        
        labels.append(labels_eval)
        
    SVM_scores = np.hstack(SVM_scores)
    SVM_znorm_scores = np.hstack(SVM_znorm_scores)
    labels = np.hstack(labels)
    
    # EDIT FOR RBF

    minDCF_SVM_score_01 = compute_minDCF(0.1, Cfn, Cfp, SVM_scores, labels)
    minDCF_SVM_znorm_score_01 = compute_minDCF(0.1, Cfn, Cfp, SVM_znorm_scores, labels)

    minDCF_SVM_score_05 = compute_minDCF(0.5, Cfn, Cfp, SVM_scores, labels)
    minDCF_SVM_znorm_score_05 = compute_minDCF(0.5, Cfn, Cfp, SVM_znorm_scores, labels)

    minDCF_SVM_score_09 = compute_minDCF(0.9, Cfn, Cfp, SVM_scores, labels)
    minDCF_SVM_znorm_score_09 = compute_minDCF(0.9, Cfn, Cfp, SVM_znorm_scores, labels)
    
    return minDCF_SVM_score_05, minDCF_SVM_znorm_score_05


features_train, labels_train = load_dataset('Train.txt')
features_train, labels_train = shuffle_dataset(features_train, labels_train)


#        SVM: linear, znorm, pca=7
#pt = 0.7
#pt = 0.3
pt=0.5
C = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10]
# svm_C_results = []
# znorm_SVM_C_results = []
# svm_PCA_C_results = []
# znorm_SVM_PCA_C_results = []
# pca_dim = 7
# P, _ = PCA(features_train, pca_dim)
# y = np.dot(P.T, features_train)
# for c in C:
#     svm_pca, znorm_svm_pca = k_fold_cv(y, labels_train, 10, "linear", c, pt)
#     svm, znorm_svm=k_fold_cv(features_train, labels_train, 10, "linear", c, pt)    
#     svm_C_results.append(svm)
#     znorm_SVM_C_results.append(znorm_svm)
#     znorm_SVM_PCA_C_results.append(znorm_svm_pca)
#     svm_PCA_C_results.append(svm_pca)
#     print(f'SVM minDCF {svm} - C = {c}')
#     print(f'ZNORM_SVM minDCF {znorm_svm} - C = {c}')
#     print(f'SVM WITH PCA=7 minDCF {svm_pca} - C = {c}')
#     print(f'ZNORM_SVM WITH PCA=7 minDCF {znorm_svm_pca} - C = {c}')

#                     ### PLOT linear SVM ###
# plt.figure(figsize=(10,6))
# plt.plot(range(len(C)), svm_C_results, color='blue', label='linear SVM')
# plt.plot(range(len(C)), svm_PCA_C_results, linestyle='-.', color='blue', label='linear SVM w/PCA=7')
# plt.plot(range(len(C)), znorm_SVM_C_results, color='red', label='linear SVM (z-norm)')
# plt.plot(range(len(C)), znorm_SVM_PCA_C_results, linestyle='-.', color='red', label='linear SVM w/PCA=7 (z-norm)')
# plt.grid()
# plt.xlabel('C')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(C)), C)
# plt.savefig('lSVM-vs-lSVM_norm-w-pca=7-pi=01-pt=05.png')
# plt.show()


# #        SVM: polynomial, znorm, pca=7
# svm_C_results = []
# znorm_SVM_C_results = []
# svm_PCA_C_results = []
# znorm_SVM_PCA_C_results = []
# pca_dim = 7
# P, _ = PCA(features_train, pca_dim)
# y = np.dot(P.T, features_train)
# for c in C:
#     svm_pca, znorm_svm_pca = k_fold_cv(y, labels_train, 10, "polynomial", c, pt)
#     svm, znorm_svm=k_fold_cv(features_train, labels_train, 10, "polynomial", c, pt)    
#     svm_C_results.append(svm)
#     znorm_SVM_C_results.append(znorm_svm)
#     znorm_SVM_PCA_C_results.append(znorm_svm_pca)
#     svm_PCA_C_results.append(svm_pca)
#     print(f'SVM polynomial minDCF {svm} - C = {c}')
#     print(f'ZNORM_SVM polynomial minDCF {znorm_svm} - C = {c}')
#     print(f'SVM WITH PCA=7 polynomial minDCF {svm_pca} - C = {c}')
#     print(f'ZNORM_SVM WITH polynomial PCA=7 minDCF {znorm_svm_pca} - C = {c}')

#                     ### PLOT polynomial SVM ###
# plt.figure(figsize=(10,6))
# plt.plot(range(len(C)), svm_C_results, color='blue', label='polynomial SVM')
# plt.plot(range(len(C)), svm_PCA_C_results, linestyle='-.', color='blue', label='polynomial SVM w/PCA=7')
# plt.plot(range(len(C)), znorm_SVM_C_results, color='red', label='polynomial SVM (z-norm)')
# plt.plot(range(len(C)), znorm_SVM_PCA_C_results, linestyle='-.', color='red', label='polynomial SVM w/PCA=7 (z-norm)')
# plt.grid()
# plt.xlabel('C')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(C)), C)
# plt.savefig('pSVM-vs-pSVM_norm-w-pca=7-pi=01-pt=05.png')
# plt.show()




####        SVM: polynomial, znorm, PCAs        ####
znorm_SVM_C_results = []
znorm_SVM_PCA7_C_results = []
znorm_SVM_PCA6_C_results = []
znorm_SVM_PCA8_C_results = []
pca_dim = 7
P, _ = PCA(features_train, 6)
y6 = np.dot(P.T, features_train)
P, _ = PCA(features_train, 7)
y7 = np.dot(P.T, features_train)
P, _ = PCA(features_train, 8)
y8 = np.dot(P.T, features_train)
for c in C:
    _, znorm_svm_pca6 = k_fold_cv(y6, labels_train, 10, "linear", c, pt)
    _, znorm_svm=k_fold_cv(features_train, labels_train, 10, "linear", c, pt)
    _, znorm_svm_pca7 = k_fold_cv(y7, labels_train, 10, "linear", c, pt)   
    _, znorm_svm_pca8 = k_fold_cv(y8, labels_train, 10, "linear", c, pt)
    znorm_SVM_C_results.append(znorm_svm)
    znorm_SVM_PCA7_C_results.append(znorm_svm_pca7)
    znorm_SVM_PCA6_C_results.append(znorm_svm_pca6)
    znorm_SVM_PCA8_C_results.append(znorm_svm_pca8)
    # print(f'SVM polynomial minDCF {svm} - C = {c}')
    # print(f'ZNORM_SVM polynomial minDCF {znorm_svm} - C = {c}')
    # print(f'SVM WITH PCA=7 polynomial minDCF {svm_pca} - C = {c}')
    # print(f'ZNORM_SVM WITH polynomial PCA=7 minDCF {znorm_svm_pca} - C = {c}')
    print(f'z-norm linear SVM - NO PCA: {znorm_svm}')
    print(f'z-norm linear SVM - PCA=6: {znorm_svm_pca6}')
    print(f'z-norm linear SVM - PCA=7: {znorm_svm_pca7}')
    print(f'z-norm linear SVM - PCA=8: {znorm_svm_pca8}')


                    ### PLOT polynomial SVM ###
plt.figure(figsize=(10,6))
plt.plot(range(len(C)), znorm_SVM_PCA6_C_results, color='blue', label='linear SVM (z-norm) w/PCA=6')
plt.plot(range(len(C)), znorm_SVM_PCA7_C_results, color='red', label='linear SVM (z-norm) w/PCA=7')
plt.plot(range(len(C)), znorm_SVM_PCA8_C_results, color='black', label='linear SVM (z-norm) w/PCA=8')
plt.plot(range(len(C)), znorm_SVM_C_results, color='brown', label='linear SVM (z-norm) NO PCA')
plt.grid()
plt.xlabel('C')
plt.ylabel('minDCF')
plt.legend()
plt.xticks(range(len(C)), C)
plt.savefig('lSVM-vs-lSVM_norm-PCA-p=05.png')
plt.show()




# #        SVM: RBF, znorm, pca=7
# svm_C_results = []
# znorm_SVM_C_results = []
# svm_PCA_C_results = []
# znorm_SVM_PCA_C_results = []
# pca_dim = 7
# P, _ = PCA(features_train, pca_dim)
# y = np.dot(P.T, features_train)
# for c in C:
#     svm_pca, znorm_svm_pca = k_fold_cv(y, labels_train, 10, "RBF", c, pt)
#     svm, znorm_svm=k_fold_cv(features_train, labels_train, 10, "RBF", c, pt)    
#     svm_C_results.append(svm)
#     znorm_SVM_C_results.append(znorm_svm)
#     znorm_SVM_PCA_C_results.append(znorm_svm_pca)
#     svm_PCA_C_results.append(svm_pca)
#     print(f'SVM RBF minDCF {svm} - C = {c}')
#     print(f'ZNORM_SVM RBF minDCF {znorm_svm} - C = {c}')
#     print(f'SVM WITH PCA=7 RBF minDCF {svm_pca} - C = {c}')
#     print(f'ZNORM_SVM WITH RBF PCA=7 minDCF {znorm_svm_pca} - C = {c}')

#                     ### PLOT SVM - RBF###
# plt.figure(figsize=(10,6))
# plt.plot(range(len(C)), svm_C_results, color='blue', label='RBF SVM')
# plt.plot(range(len(C)), svm_PCA_C_results, linestyle='-.', color='blue', label='RBF SVM w/PCA=7')
# plt.plot(range(len(C)), znorm_SVM_C_results, color='red', label='RBF SVM (z-norm)')
# plt.plot(range(len(C)), znorm_SVM_PCA_C_results, linestyle='-.', color='red', label='RBF SVM w/PCA=7 (z-norm)')
# plt.grid()
# plt.xlabel('C')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(C)), C)
# plt.savefig('rbfSVM-vs-rbfSVM_norm-w-pca=7-pi=01-pt=05.png')
# plt.show()



#        SVM: RBF pca with different gamma values

# znorm_SVM_results = []
# znorm_SVM_results_gamma2 = []
# znorm_SVM_results_gamma3 = []
# znorm_SVM_results_gamma4 = []
# znorm_SVM_results_gamma5 = []

# pca_dim = 7
# P, _ = PCA(features_train, pca_dim)
# y = np.dot(P.T, features_train)
# for c in C:
#     _, znorm_svm_pca = k_fold_cv(y, labels_train, 10, "RBF", c, pt)
#     _, znorm_svm_pca_gamma2 = k_fold_cv(y, labels_train, 10, "RBF", c, pt, gamma=0.1)
#     _, znorm_svm_pca_gamma3 = k_fold_cv(y, labels_train, 10, "RBF", c, pt, gamma=0.01)
#     _, znorm_svm_pca_gamma4 = k_fold_cv(y, labels_train, 10, "RBF", c, pt, gamma=0.001)
# #    _, znorm_svm_pca_gamma5 = k_fold_cv(y, labels_train, 10, "RBF", c, pt, gamma=0.0001)
   
#     znorm_SVM_results.append(znorm_svm_pca)
#     znorm_SVM_results_gamma2.append(znorm_svm_pca_gamma2)
#     znorm_SVM_results_gamma3.append(znorm_svm_pca_gamma3)
#     znorm_SVM_results_gamma4.append(znorm_svm_pca_gamma4)
# #    znorm_SVM_results_gamma5.append(znorm_SVM_results_gamma5)
#     print(f'C: {c}')
#     print(f'SVM RBF - Gamma=1: {znorm_svm_pca}')
#     print(f'SVM RBF - Gamma=0.1: {znorm_svm_pca_gamma2}')
#     print(f'SVM RBF - Gamma=0.01: {znorm_svm_pca_gamma3}')
#     print(f'SVM RBF - Gamma=0.001: {znorm_svm_pca_gamma4}')
#     #print(f'SVM RBF - Gamma=0.0001: {znorm_svm_pca_gamma5}')



#                     ### PLOT SVM - RBF###
# plt.figure(figsize=(10,6))
# plt.plot(range(len(C)), znorm_SVM_results, color='blue', label='RBF SVM w/PCA=7 (z-norm) gamma=1')
# plt.plot(range(len(C)), znorm_SVM_results_gamma2, color='red',label='RBF SVM w/PCA=7 (z-norm) gamma=0.1')
# plt.plot(range(len(C)), znorm_SVM_results_gamma3, color='black', label='RBF SVM w/PCA=7 (z-norm) gamma=0.01')
# plt.plot(range(len(C)), znorm_SVM_results_gamma4, color='green', label='RBF SVM w/PCA=7 (z-norm) gamma=0.001')
# #plt.plot(range(len(C)), znorm_SVM_results_gamma5, color='cyan', label='RBF SVM w/PCA=7 (z-norm) gamma=0.0001')
# plt.grid()
# plt.xlabel('C')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(C)), C)
# plt.savefig('rbfSVM-vs-rbfSVM_norm-w-pca=7-pi=05-pt=05-gamma.png')
# plt.show()




#        SVM: RBF, znorm, pca=7
# svm01_PCA_C_results = []
# znorm01_SVM_PCA_C_results = []

# svm05_PCA_C_results = []
# znorm05_SVM_PCA_C_results = []

# svm09_PCA_C_results = []
# znorm09_SVM_PCA_C_results = []

# pca_dim = 7
# P, _ = PCA(features_train, pca_dim)
# y = np.dot(P.T, features_train)
# for c in C:
#     svm_pca01, znorm_svm_pca01, svm_pca05, znorm_svm_pca05, svm_pca09, znorm_svm_pca09  = k_fold_cv(y, labels_train, 10, "RBF", c, pt, gamma=0.01)

#     znorm01_SVM_PCA_C_results.append(znorm_svm_pca01)
#     svm01_PCA_C_results.append(svm_pca01)

#     znorm05_SVM_PCA_C_results.append(znorm_svm_pca05)
#     svm05_PCA_C_results.append(svm_pca05)

#     znorm09_SVM_PCA_C_results.append(znorm_svm_pca09)
#     svm09_PCA_C_results.append(svm_pca09)
#     print(f'pi=01 - SVM WITH PCA=7 RBF minDCF {svm_pca01} - C = {c}')
#     print(f'pi=01 ZNORM_SVM WITH RBF PCA=7 minDCF {znorm_svm_pca01} - C = {c}')
#     print(f'pi=05 - SVM WITH PCA=7 RBF minDCF {svm_pca05} - C = {c}')
#     print(f'pi=05 ZNORM_SVM WITH RBF PCA=7 minDCF {znorm_svm_pca05} - C = {c}')
#     print(f'pi=09 - SVM WITH PCA=7 RBF minDCF {svm_pca09} - C = {c}')
#     print(f'pi=09 ZNORM_SVM WITH RBF PCA=7 minDCF {znorm_svm_pca09} - C = {c}')
#                     ### PLOT SVM - RBF###
# plt.figure(figsize=(10,6))
# plt.plot(range(len(C)), svm01_PCA_C_results, color='blue', label='RBF SVM w/PCA=7, pi=01')
# plt.plot(range(len(C)), svm05_PCA_C_results, color='red', label='RBF SVM w/PCA=7, pi=05')
# plt.plot(range(len(C)), svm09_PCA_C_results, color='green', label='RBF SVM w/PCA=7, pi=09')
# plt.plot(range(len(C)), znorm01_SVM_PCA_C_results, linestyle='-.', color='blue', label='RBF SVM w/PCA=7 (z-norm), pi=01')
# plt.plot(range(len(C)), znorm05_SVM_PCA_C_results, linestyle='-.', color='red', label='RBF SVM w/PCA=7 (z-norm), pi=05')
# plt.plot(range(len(C)), znorm09_SVM_PCA_C_results, linestyle='-.', color='green', label='RBF SVM w/PCA=7 (z-norm), pi=09')
# plt.grid()
# plt.xlabel('C')
# plt.ylabel('minDCF')
# plt.legend()
# plt.xticks(range(len(C)), C)
# plt.savefig('rbf-pt=05-different-pi-gamma=001.png')
# plt.show()



