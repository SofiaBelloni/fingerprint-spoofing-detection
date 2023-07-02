import numpy as np
from project import covariance_matrix

def vcol(v):
    return v.reshape(v.size, 1)

def vrow(v):
    return v.reshape(1, v.size)

def compute_ML_estimates(features, labels):
    total_sigma = []
    total_mu = []
    for c in set(labels):
        feat_c = features[:, labels==c]
        total_sigma.append(covariance_matrix(feat_c))
        total_mu.append(vcol(np.mean(feat_c)))

    return np.array(total_sigma), np.array(total_mu)

def logpdf_GAU_ND_Array(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = - 0.5 * M * np.log(2*np.pi)
    logdet = np.linalg.slogdet(C)[1] #restituisce due valori, io voglio il secondo (il primo è il segno)
    L = np.linalg.inv(C)
    v = (XC * np.dot(L, XC)).sum(0)
    return const -0.5 *logdet - 0.5 * v

def gaussian_classifier(features_train, labels_train, features_test):
    sigma, mu = compute_ML_estimates(features_train, labels_train)

    print(sigma.shape)
    #  **** CLASSIFICATION ****

    # first step: compute, for each test sample, the likelihoods
    # Store class-conditional probabilities in a score matrix S. 
    # Each row of the score matrix corresponds to a class, and
    # contains the conditional log-likelihoods for all the samples for that class.
    S = []
    for c in set(labels_train):
        print(c)
        s = sigma[c, :, :]
        m = mu[c, :]
        fcond = np.exp(logpdf_GAU_ND_Array(features_test, m, s))
        S.append(fcond)

    #Compute class posterior probabilities combining the score matrix with prior information.
    p = vcol(np.ones(len(S))/2)
    SJoint = S * p
    SMarginal = vrow(SJoint.sum(0))
    Post_prob = SJoint / SMarginal
    #The predicted label is obtained as the class that has maximum posterior probability.
    return np.argmax(Post_prob, axis=0)




    



    
