from project import *
import MVG_classifiers
from logistic_regression import logRegClass, quadratic_features_expansion

### LOADING THE DATASET ###
features_train, labels_train = load_dataset('Train.txt')
features_test, labels_test = load_dataset('Test.txt')

### PRINTING THE DATASET ###
print_dataset_info(features_train, labels_train)
print_dataset_info(features_test, labels_test)

### PLOTTING THE DATASET ###
#plot_histogram(features_train, labels_train, 'hist')
#plot_scatter(features_train, labels_train)

###PCA###
P = PCA(features_train)
y = np.dot(P.T, features_train)
#plot_histogram(y, labels_train, 'hist-pca')
#scatter_pca(y, labels_train)

###LDA###
W = LDA(features_train, labels_train, m=1)
y_lda = np.dot(W.T, features_train)
#plot_histogram(y_lda, labels_train, 'hist-lda')

###Pearson correlations###
#compute_pearson_correlation(features_train, 'Greys', 'dataset-corr')
#compute_pearson_correlation(features_train[:, labels_train==1], 'Blues', 'authentical-corr')
#comput.e_pearson_correlation(features_train[:, labels_train==0], 'Reds', 'spoofed-corr')

### MVG ###
labels_MVG = MVG_classifiers.gaussian_classifier(features_train, labels_train, features_test)
accuracy = np.sum(labels_MVG == labels_test) / labels_MVG.shape[0]
#print(labels_MVG)
#print(labels_test)
print(f'MVG Classifier standard ACCURACY: {accuracy * 100}%')

labels_MVG_nb = MVG_classifiers.nb_gaussian_classifier(features_train, labels_train, features_test)
accuracy = np.sum(labels_MVG_nb == labels_test) / labels_MVG_nb.shape[0]
print(f'MVG NB Classifier ACCURACY: {accuracy * 100}%')

labels_MVG_tied = MVG_classifiers.tied_classifier(features_train, labels_train, features_test)
accuracy = np.sum(labels_MVG_tied == labels_test) / labels_MVG_tied.shape[0]
print(f'MVG TIED Classifier ACCURACY: {accuracy * 100}%')


#### LOGISTIC REGRESSION ####
logRegMethod = logRegClass(features_train, labels_train, 0.001)
predictions = logRegMethod.predict(features_test)
accuracy = (predictions==labels_test).sum()/predictions.shape[0]
print(f'LOGISTIC REGRESSION Classifier ACCURACY: {accuracy * 100}%')

#### LOGISTIC REGRESSION WITH QUADRATIC FEATURES EXPANSION ####
features_expanded, labels_expanded = quadratic_features_expansion(features_train, labels_train)
ftest_expanded, ltest_expanded = quadratic_features_expansion(features_train, labels_train)
logRegMethod = logRegClass(features_expanded, labels_expanded, 0.001)
predictions = logRegMethod.predict(ftest_expanded)
accuracy = (predictions==ltest_expanded).sum()/predictions.shape[0]
print(f'LOGISTIC REGRESSION WITH QUADRATIC FEATURES EXPANSION Classifier ACCURACY: {accuracy * 100}%')