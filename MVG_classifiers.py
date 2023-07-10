import numpy as np
import scipy
from project import * 



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

    #  **** CLASSIFICATION ****

    # first step: compute, for each test sample, the likelihoods
    # Store class-conditional probabilities in a score matrix S. 
    # Each row of the score matrix corresponds to a class, and
    # contains the conditional log-likelihoods for all the samples for that class.
    S = []
    for c in set(labels_train):
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

def nb_gaussian_classifier(features_train, labels_train, features_test):
    sigma, mu = compute_ML_estimates(features_train, labels_train)
    S = []
    for c in set(labels_train):
        s = sigma[c, :, :]
        s = s * np.identity(s.shape[0])
        m = mu[c, :]
        fcond = np.exp(logpdf_GAU_ND_Array(features_test, m, s))
        S.append(fcond)
    p = vcol(np.ones(len(S)) * np.log(1/2))
    logSJoint = S + p
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logPost_prob = logSJoint - logSMarginal
    Post_prob = np.exp(logPost_prob)
    return np.argmax(Post_prob, axis=0)

def tied_classifier(features_train, labels_train, features_test):
    sigma = within_class_covariance_matrix(features_train, labels_train)
    total_mu = []
    for c in set(labels_train):
        feat_c = features_train[:, labels_train==c]
        total_mu.append(vcol(np.mean(feat_c)))
    total_mu = np.array(total_mu)
    S = []
    for c in set(labels_train):
        m = total_mu[c, :]
        fcond = np.exp(logpdf_GAU_ND_Array(features_test, m, sigma))
        S.append(fcond)
    p = vcol(np.ones(len(S)) * np.log(1/2))
    logSJoint = S + p
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logPost_prob = logSJoint - logSMarginal
    Post_prob = np.exp(logPost_prob)
    return np.argmax(Post_prob, axis=0)







    



    
