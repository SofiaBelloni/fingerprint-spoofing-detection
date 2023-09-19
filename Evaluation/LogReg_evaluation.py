import sys
sys.path.append('C:\\Users\\sofia\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from logistic_regression import *
from project import *
from metrics import *

# this function return minDCF computed with LogReg
def log_reg(f_train, l_train, f_test, l_test, l, pt):
    logreg = logRegClass(f_train, l_train, l, pt)
    logReg_scores = logreg.compute_scores(f_test)
    minDCF_LogReg = compute_minDCF(pi, Cfn, Cfp, logReg_scores, l_test)
    print(f'minDCF l = {l}:')
    print(minDCF_LogReg)
    return minDCF_LogReg

### LOADING THE DATASET ###
print(f'APPLICATION (%.1f, %d, %d)'% (pi, Cfn, Cfp))
features_train, labels_train = load_dataset('Train.txt')
features_test, labels_test = load_dataset('Test.txt')
print('Train')
#print_dataset_info(features_train, labels_train)
print('Test')
#print_dataset_info(features_test, labels_test)

#### Apply PCA e znormalization ####
pca = 7
N = 2
P, _ = PCA(features_train, pca)
y_train = np.dot(P.T, features_train)

y_test = np.dot(P.T, features_test)

znorm_train_pca, znorm_test_pca = znorm(y_train, y_test)
znorm_train, znorm_test = znorm(features_train, features_test)



#### COMPUTE minDCF with LogReg ####
plt.figure(figsize=(10,6))
logreg_lambda_results = []
logreg_lambda_results_pca = []
znorm_logreg_lambda_results = []
znorm_logreg_lambda_results_pca = []
lam = [1E-4, 1E-3, 1E-2, 1E-1]
print('Logistic Regression:')
for l in lam:
    print('Raw feature')
    minDCF_LogReg = log_reg(features_train, labels_train, features_test, labels_test, l, None)
    logreg_lambda_results.append(minDCF_LogReg)
    print('PCA')
    minDCF_LogReg_pca = log_reg(y_train, labels_train, y_test, labels_test, l, None)
    logreg_lambda_results_pca.append(minDCF_LogReg_pca)
    print('Znorm')
    minDCF_LogReg_znorm = log_reg(znorm_train, labels_train, znorm_test, labels_test, l, None)
    znorm_logreg_lambda_results.append(minDCF_LogReg_znorm)
    print('Znorm PCA')
    minDCF_LogReg_znorm_pca = log_reg(znorm_train_pca, labels_train, znorm_test_pca, labels_test, l, None)
    znorm_logreg_lambda_results_pca.append(minDCF_LogReg_znorm_pca)

plt.plot(range(len(lam)), logreg_lambda_results, color='blue', label=f'LogReg')
plt.plot(range(len(lam)), logreg_lambda_results_pca, color='red', label=f'LogReg - PCA = 7')
plt.plot(range(len(lam)), znorm_logreg_lambda_results, color='black', label=f'LogReg znorm')
plt.plot(range(len(lam)), znorm_logreg_lambda_results_pca, color='green', label=f'LogReg znorm - PCA = 7')
plt.grid()
plt.xlabel('λ')
plt.ylabel('minDCF')
plt.legend()
plt.xticks(range(len(lam)), lam)
plt.savefig('logreg-eval.png')
plt.show()


#### COMPUTE minDCF with Prior Weighted LogReg ####
pt = 0.3

plt.figure(figsize=(10,6))
logreg_lambda_results = []
logreg_lambda_results_pca = []
znorm_logreg_lambda_results = []
znorm_logreg_lambda_results_pca = []
print('Prior Weighted Logistic Regression:')
for l in lam:
    print('Raw feature')
    minDCF_LogReg = log_reg(features_train, labels_train, features_test, labels_test, l, pt)
    logreg_lambda_results.append(minDCF_LogReg)
    print('PCA')
    minDCF_LogReg_pca = log_reg(y_train, labels_train, y_test, labels_test, l, pt)
    logreg_lambda_results_pca.append(minDCF_LogReg_pca)
    print('Znorm')
    minDCF_LogReg_znorm = log_reg(znorm_train, labels_train, znorm_test, labels_test, l, pt)
    znorm_logreg_lambda_results.append(minDCF_LogReg_znorm)
    print('Znorm PCA')
    minDCF_LogReg_znorm_pca = log_reg(znorm_train_pca, labels_train, znorm_test_pca, labels_test, l, pt)
    znorm_logreg_lambda_results_pca.append(minDCF_LogReg_znorm_pca)

plt.plot(range(len(lam)), logreg_lambda_results, color='blue', label=f'PWLogReg')
plt.plot(range(len(lam)), logreg_lambda_results_pca, color='red', label=f'PWLogReg - PCA = 7')
plt.plot(range(len(lam)), znorm_logreg_lambda_results, color='black', label=f'PWLogReg znorm')
plt.plot(range(len(lam)), znorm_logreg_lambda_results_pca, color='green', label=f'PWLogReg znorm - PCA = 7')
plt.grid()
plt.xlabel('λ')
plt.ylabel('minDCF')
plt.legend()
plt.xticks(range(len(lam)), lam)
plt.savefig('PWLogReg-eval.png')
plt.show()