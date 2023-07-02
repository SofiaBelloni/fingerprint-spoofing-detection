from project import *
import MVG_classifiers

### LOADING THE DATASET ###
features_train, labels_train = load_dataset('Train.txt')
features_test, labels_test = load_dataset('Test.txt')

### PRINTING THE DATASET ###
print_dataset_info(features_train, labels_test)
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
print(labels_MVG)
print(labels_test)
print(f'MVG Classifier ACCURACY: {accuracy}')