import numpy as np
import matplotlib.pyplot as plt
def load_dataset(filename):
    samples = []
    labels = []
    with open(filename) as f:
        for line in f:
            splitted = line.split(",")
            sample = [float(i) for i in splitted[0:10]]
            label = float(splitted[10].strip())
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

def covariance_matrix(features):
    # Compute mean
    x_bar = np.mean(features, 1).reshape(features.shape[0], 1)
    # Center data
    z = features - x_bar
    # Compute covariance matrix
    C = np.dot(z, z.T) / features.shape[1]
    return C


def PCA(features, m = 2):
    #1. Compute covariance matrix
    C = covariance_matrix(features)
    #2. Took the eigenvectors
    _, U = np.linalg.eigh(C)
    P = U[:,::-1][:,0:m]
    return P

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
plot_histogram(y, labels_train, 'hist-pca')
scatter_pca(y, labels_train)









    
