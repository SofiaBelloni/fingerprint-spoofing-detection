import sys
sys.path.append('C:\\Users\\sofia\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from project import *
from MVG_classifier import MVG
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
P, _ = PCA(features_train, pca)
y_train = np.dot(P.T, features_train)

y_test = np.dot(P.T, features_test)

znorm_train_pca, znorm_test_pca = znorm(y_train, y_test)
znorm_train, znorm_test = znorm(features_train, features_test)

#### COMPUTE minDCF no pca ####
print('No PCA:')
mvg = MVG(znorm_train, labels_train)
print('Computing MVG standard scores...')
MVG_scores = mvg.llr(znorm_test, 'standard')
print('Computing MVG nb scores...')
MVG_NB_scores = mvg.llr(znorm_test, 'nb')

minDCF_MVG_score = compute_minDCF(pi, Cfn, Cfp, MVG_scores, labels_test)
minDCF_MVG_NB_score = compute_minDCF(pi, Cfn, Cfp, MVG_NB_scores, labels_test)

print('minDCF MVG:')
print(minDCF_MVG_score)
print('minDCF MVG nb:')
print(minDCF_MVG_NB_score)

#### COMPUTE minDCF with pca = 7 ####
print('With PCA = 7:')
mvg = MVG(znorm_train_pca, labels_train)
print('Computing MVG standard scores...')
MVG_scores = mvg.llr(znorm_test_pca, 'standard')
print('Computing MVG nb scores...')
MVG_NB_scores = mvg.llr(znorm_test_pca, 'nb')

minDCF_MVG_score = compute_minDCF(pi, Cfn, Cfp, MVG_scores, labels_test)
minDCF_MVG_NB_score = compute_minDCF(pi, Cfn, Cfp, MVG_NB_scores, labels_test)

print('minDCF MVG:')
print(minDCF_MVG_score)
print('minDCF MVG nb:')
print(minDCF_MVG_NB_score)

