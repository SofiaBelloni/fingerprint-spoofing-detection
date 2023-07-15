import numpy as np
import scipy
from project import vcol, vrow

class SupportVectorMachines:
    def __init__(self, DTR, LTR, C, pT, gamma=1, d=2, K=1):
        self.DTR = DTR
        self.LTR = LTR
        self.C = C
        self.pT = pT
        self.d = d
        self.gamma = gamma
        self.K = K
        self.w_start = None
        self.H = None
    
    def train(self, mode):
        DTRext = np.vstack([self.DTR, np.ones((1, self.DTR.shape[1]))])
        
        DTR0 = self.DTR[:, self.LTR==0]
        DTR1 = self.DTR[:, self.LTR==1]
        nF = DTR0.shape[1]
        nT = DTR1.shape[1]
        emp_prior_F = (nF / self.DTR.shape[1])
        emp_prior_T =  (nT / self.DTR.shape[1])
        Cf = self.C * self.pT / emp_prior_F
        Ct = self.C * self.pT / emp_prior_T
    
        Z = np.zeros(self.LTR.shape)
        Z[self.LTR == 0] = -1
        Z[self.LTR == 1] = 1
        
        if mode == "linear":
            H = np.dot(DTRext.T, DTRext)
            H = vcol(Z) * vrow(Z) * H
        elif mode == "polynomial":
            H = np.dot(DTRext.T, DTRext) ** self.d
            H = vcol(Z) * vrow(Z) * H
        elif mode == "RBF":
            dist = vcol((self.DTR**2).sum(0)) + vrow((self.DTR**2).sum(0)) - 2*np.dot(self.DTR.T, self.DTR)
            H = np.exp(-self.gamma * dist) + self.K
            H = vcol(Z) * vrow(Z) * H
        
        self.H = H
        
        bounds = [(-1, -1)] * self.DTR.shape[1]
        for i in range(self.DTR.shape[1]):
            if self.LTR[i] == 0:
                bounds[i] = (0, Cf)
            else:
                bounds[i] = (0, Ct)
        
        alpha_star, x, y = scipy.optimize.fmin_l_bfgs_b(
            self._LDual, 
            np.zeros(self.DTR.shape[1]),
            #bounds = [(0, self.C)] * DTR.shape[1],
            bounds = bounds,
            factr = 1e7,
            maxiter = 100000,
            maxfun = 100000
                )

        self.w_star = np.dot(DTRext, vcol(alpha_star) * vcol(Z))
    
    def compute_scores(self, DTE, mode):
        self.train(mode)
        DTEext = np.vstack([DTE, np.ones((1, DTE.shape[1]))])
        S = np.dot(self.w_star.T, DTEext).ravel()
        return S
        
    def _JDual(self, alpha):
        Ha = np.dot(self.H, vcol(alpha))
        aHa = np.dot(vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
    def _LDual(self, alpha):
        loss, grad = self._JDual(alpha)
        return -loss, -grad
    
    def _JPrimal(self, DTRext, w, Z):
        S = np.dot(vrow(w), DTRext)
        loss = np.maximum(np.zeros(S.shape), 1-Z*S).sum()
        return 0.5*np.linalg.norm(w)**2 + self.C*loss