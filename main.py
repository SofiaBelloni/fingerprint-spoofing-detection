from project import *
from MVG_classifier import MVG
from logistic_regression import logRegClass, quadratic_features_expansion
from SVM_classifier import SupportVectorMachines
from GMM_classifier import GMM_classifier
from metrics import *

### LOADING THE DATASET ###
features_train, labels_train = load_dataset('Train.txt')
features_test, labels_test = load_dataset('Test.txt')
print('Train')
print_dataset_info(features_train, labels_train)
print('Test')
print_dataset_info(features_test, labels_test)
### PLOTTING PCA EXPLAINED VARIANCE
plot_explained_variance_pca(features_train)

pi = 0.5
Cfn = 1
Cfp = 10

#### COMPUTE minDCF ####
mvg = MVG(features_train, labels_train)
MVG_scores = mvg.llr(features_test, 'standard')
print(MVG_scores.shape)
#MVG_NB_scores = mvg.llr(features_test, 'nb')
#MVG_TIED_scores = mvg.llr(features_test, 'tied')

#minDCF_MVG_score = compute_minDCF(pi, Cfn, Cfp, MVG_scores, labels_test)
#minDCF_MVG_NB_score = compute_minDCF(pi, Cfn, Cfp, MVG_NB_scores, labels_test)
#minDCF_MVG_TIED_score = compute_minDCF(pi, Cfn, Cfp, MVG_TIED_scores, labels_test)

#print(minDCF_MVG_score)

#print(minDCF_MVG_NB_score)
#print(minDCF_MVG_TIED_score)

