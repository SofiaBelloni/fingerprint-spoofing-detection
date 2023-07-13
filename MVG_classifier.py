import numpy as np
import scipy
from project import * 

class MVG:
    def __init__(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR

    def compute_ML_estimates(self):
        total_sigma = []
        total_mu = []
        for c in set(self.LTR):
            feat_c = self.DTR[:, self.LTR==c]
            total_sigma.append(covariance_matrix(feat_c))
            total_mu.append(vcol(np.mean(feat_c, axis=1)))
        return np.array(total_sigma), np.array(total_mu)

    def logpdf_GAU_ND_Array(self, X, mu, C):
        XC = X - mu
        M = X.shape[0]
        const = - 0.5 * M * np.log(2*np.pi)
        logdet = np.linalg.slogdet(C)[1] #restituisce due valori, io voglio il secondo (il primo è il segno)
        L = np.linalg.inv(C)
        v = (XC * np.dot(L, XC)).sum(0)
        return const -0.5 *logdet - 0.5 * v
    

    def logLikelihoodScore(self, DTE, mode):
        S = []
        if mode == 'nb':
            sigma, mu = self.compute_ML_estimates()
            for c in set(self.LTR):
                s = sigma[c, :, :]
                s = s * np.identity(s.shape[0])
                m = mu[c, :]
                fcond = self.logpdf_GAU_ND_Array(DTE, m, s)
                S.append(fcond)

        elif mode == 'tied':
            sigma = within_class_covariance_matrix(self.DTR, self.LTR)
            total_mu = []
            for c in set(self.LTR):
                feat_c = self.DTR[:, self.LTR==c]
                total_mu.append(vcol(np.mean(feat_c, axis=1)))
            total_mu = np.array(total_mu)
            for c in set(self.LTR):
                m = total_mu[c, :]
                fcond = self.logpdf_GAU_ND_Array(DTE, m, sigma)
                S.append(fcond)

        elif mode == 'standard':
            sigma, mu = self.compute_ML_estimates()
            for c in set(self.LTR):
                s = sigma[c, :, :]
                m = mu[c, :]
                fcond = self.logpdf_GAU_ND_Array(DTE, m, s).ravel()
                S.append(fcond)
        else:
            exit("error mode")
        return S
    
    def llr(self, DTE, mode):
        S = self.logLikelihoodScore(DTE, mode)
        return S[1] - S[0]
    
    def predict(self, DTE, mode):
        S = self.logLikelihoodScore(DTE, mode)
        p = vcol(np.ones(len(S)) * np.log(1/2))
        logSJoint = S + p
        logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logPost_prob = logSJoint - logSMarginal 
        Post_prob = np.exp(logPost_prob)
        return np.argmax(Post_prob, axis=0)

