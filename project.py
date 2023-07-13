import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import seaborn

pi = 0.5
Cfn = 1
Cfp = 10

def vcol(v):
    return v.reshape(v.size, 1)

def vrow(v):
    return v.reshape(1, v.size)

def shuffle_dataset(features_train, labels_train):
    total_trainset = np.vstack((features_train, labels_train))
    np.random.shuffle(total_trainset.T)
    features_train = total_trainset[0:-1, :]
    labels_train = np.int32(total_trainset[-1, :])
    return features_train, labels_train

def load_dataset(filename):
    samples = []
    labels = []
    with open(filename) as f:
        for line in f:
            splitted = line.split(",")
            sample = [float(i) for i in splitted[0:10]]
            label = int(splitted[10].strip())
            samples.append(sample)
            labels.append(label)
    # Forse dobbiamo trasporre i samples, ci serve un vettore colonna.
    return np.array(samples).T, np.array(labels)

def get_num_labels(labels, value):
    count=0
    for elem in labels:
        if(elem == value):
            count = count + 1
    return count

def plot_histogram(features, labels, name):
    feat_0 = features[:, labels==0] 
    feat_1 = features[:, labels==1]
    for i in range(0,features.shape[0]):
        plt.figure()
        plt.hist(feat_0[i], bins=10, density=True, alpha=0.4, color='red', label='Spoofed')
        plt.hist(feat_1[i], bins=10, density=True, alpha=0.4, color='blue', label='Authentical')
        plt.legend()
        plt.tight_layout()
        plt.savefig('.\Histogram\%s_%d.png' % (name,i) )
        #plt.show()

def plot_scatter(features, labels):
    feat_0 = features[:, labels==0] 
    feat_1 = features[:, labels==1]
    for i in range(0,10):
        for j in range(0,10):
            plt.figure()
            plt.scatter(feat_0[i, :], feat_0[j, :], label='Spoofed')
            plt.scatter(feat_1[i, :], feat_1[j, :], label='Authentical')
            plt.legend()
            plt.tight_layout()
            plt.savefig('.\Scatter\scatter_%d-%d.png' % (i, j))

def compute_pearson_correlation(features, color, title):
    corr = np.corrcoef(features)
    plt.figure()
    seaborn.heatmap(corr, cmap=color)
    #plt.show()
    plt.savefig('.\Correlation\%s.png' % title)
    print('Pearson Correlation')

def covariance_matrix(features):
    # Compute mean
    x_bar = np.mean(features, 1).reshape(features.shape[0], 1)
    # Center data
    z = features - x_bar
    # Compute covariance matrix
    C = np.dot(z, z.T) / features.shape[1]
    return C

def between_class_covariance_matrix(features, labels):
    C_matrices = []
    mu = np.mean(features, 1).reshape(features.shape[0], 1)
    for i in set(labels):
        feat_c = features[:, labels==i]
        mu_c = np.mean(feat_c, 1).reshape(feat_c.shape[0], 1)
        diff = mu_c - mu
        C_matrices.append(np.dot(diff, diff.T) * feat_c.shape[1])
    
    C_matrices = np.array(C_matrices)
    return C_matrices.sum(0) / features.shape[1]

def within_class_covariance_matrix(features, labels):
    C_matrices = []
    for i in set(labels):
        feat_c = features[:, labels==i]
        S_Wc = covariance_matrix(feat_c)
        C_matrices.append(S_Wc * feat_c.shape[1])
    C_matrices = np.array(C_matrices)
    return C_matrices.sum(0) / features.shape[1]


def PCA(features, m = 2):
    #1. Compute covariance matrix
    C = covariance_matrix(features)
    #2. Took the eigenvectors
    eigenvalues, U = np.linalg.eigh(C)
    explained_variance = eigenvalues / np.sum(eigenvalues)
    P = U[:,::-1][:,0:m]
    return P, explained_variance

def LDA(features, labels, m=2):
    # Compute between class covariance
    S_b = between_class_covariance_matrix(features, labels)
    #Compute within class covariance
    S_w = within_class_covariance_matrix(features, labels)
    #Compute eigenvector
    _, U = scipy.linalg.eigh(S_b, S_w)
    W = U[:,::-1][:,0:m]
    return W

def print_dataset_info(features, labels):
    print('Features shape')
    print(features.shape)
    print('Labels shape')
    print(labels.shape)
    print('Labels shape that are 1')
    print(get_num_labels(labels, 1))
    print('Labels shape that are 0')
    print(get_num_labels(labels, 0))

def scatter_pca(features, labels):
    feat_0 = features[:, labels==0] 
    feat_1 = features[:, labels==1]
    plt.figure()
    plt.scatter(feat_0[0], feat_0[1], label='Spoofed')
    plt.scatter(feat_1[0], feat_1[1], label='Authentical')
    plt.savefig('.\Scatter\scatter-pca.png')
    #plt.show()

def plot_explained_variance_pca(features):

    _, explained_variance = PCA(features)
    sorted_indices = np.argsort(explained_variance)[::-1]
    sorted_explained_variance = explained_variance[sorted_indices]
    cumulative_variance = np.cumsum(sorted_explained_variance)
    # Plot della varianza spiegata
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-o', label='Varianza cumulativa')
    plt.xlabel('PCA dimensions')
    plt.ylabel('Fraction of explained variance')
    plt.grid(color='grey')
    plt.xticks(range(11))
    plt.show()









    
