import sys
sys.path.append('C:\\Users\\sofia\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from project import *
from SVM_classifier import SupportVectorMachines
from metrics import *

### LOADING THE DATASET ###
print(f'APPLICATION (%.1f, %d, %d)'% (pi, Cfn, Cfp))
features_train, labels_train = load_dataset('Train.txt')
features_test, labels_test = load_dataset('Test.txt')
print('Train')
print_dataset_info(features_train, labels_train)
print('Test')
print_dataset_info(features_test, labels_test)

#### Apply PCA e znormalization ####
pca = 7
N = 2
P, _ = PCA(features_train, pca)
y_train = np.dot(P.T, features_train)

y_test = np.dot(P.T, features_test)

znorm_train_pca, znorm_test_pca = znorm(y_train, y_test)
znorm_train, znorm_test = znorm(features_train, features_test)

#### COMPUTE minDCF no pca ####
print('No PCA:')
gmm = GMM_classifier(znorm_train, labels_train, N)
print('Computing GMM diagonal tied scores...')
GMM_naive_tied_scores = gmm.compute_scores(znorm_test, 'naive', 'tied')

minDCF_GMM = compute_minDCF(pi, Cfn, Cfp, GMM_naive_tied_scores, labels_test)

print('minDCF:')
print(minDCF_GMM)

#### COMPUTE minDCF with pca = 7 ####
print('With PCA = 7:')
gmm = GMM_classifier(znorm_train_pca, labels_train, N)
print('Computing GMM diagonal tied scores...')
GMM_naive_tied_scores = gmm.compute_scores(znorm_test_pca, 'naive', 'tied')
minDCF_GMM = compute_minDCF(pi, Cfn, Cfp, GMM_naive_tied_scores, labels_test)

print('minDCF:')
print(minDCF_GMM)