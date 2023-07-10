import numpy
import scipy
from project import vcol, vrow

class SupportVectorMachines:
    def __init__(self, DTR, LTR, C, mode, pT, gamma=1, d=2, K=1):
        self.DTR = DTR
        self.LTR = LTR
        self.C = C
        self.mode = mode
        self.pT = pT
        self.d = d
        self.gamma = gamma
        self.K = K
        self.w_start = None
        self.H = None
    
    def train(self):
        DTRext = numpy.vstack([self.DTR, numpy.ones((1, self.DTR.shape[1]))])
        
        DTR0 = self.DTR[:, self.LTR==0]
        DTR1 = self.DTR[:, self.LTR==1]
        nF = DTR0.shape[1]
        nT = DTR1.shape[1]
        emp_prior_F = (nF / self.DTR.shape[1])
        emp_prior_T =  (nT / self.DTR.shape[1])
        Cf = self.C * self.pT / emp_prior_F
        Ct = self.C * self.pT / emp_prior_T
    
        Z = numpy.zeros(self.LTR.shape)
        Z[self.LTR == 0] = -1
        Z[self.LTR == 1] = 1
        
        if self.mode == "linear":
            H = numpy.dot(DTRext.T, DTRext)
            H = vcol(Z) * vrow(Z) * H
        elif self.mode == "polynomial":
            H = numpy.dot(DTRext.T, DTRext) ** self.d
            H = vcol(Z) * vrow(Z) * H
        elif self.mode == "RBF":
            dist = vcol((self.DTR**2).sum(0)) + vrow((self.DTR**2).sum(0)) - 2*numpy.dot(self.DTR.T, self.DTR)
            H = numpy.exp(-self.gamma * dist) + self.K
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
            numpy.zeros(self.DTR.shape[1]),
            #bounds = [(0, self.C)] * DTR.shape[1],
            bounds = bounds,
            factr = 1e7,
            maxiter = 100000,
            maxfun = 100000
                )

        self.w_star = numpy.dot(DTRext, vcol(alpha_star) * vcol(Z))

    def temporary_predict(self, DTE):
        self.train()
        DTEext = numpy.vstack([DTE, numpy.ones((1, DTE.shape[1]))])
        S = numpy.dot(self.w_star.T, DTEext)
        predictions = S > 0
        return predictions
    
    def compute_scores(self, DTE):
        DTEext = numpy.vstack([DTE, numpy.ones((1, DTE.shape[1]))])
        S = numpy.dot(self.w_star.T, DTEext)
        return S
        
    def _JDual(self, alpha):
        Ha = numpy.dot(self.H, vcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)
    
    def _LDual(self, alpha):
        loss, grad = self._JDual(alpha)
        return -loss, -grad
    
    def _JPrimal(self, DTRext, w, Z):
        S = numpy.dot(vrow(w), DTRext)
        loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
        return 0.5*numpy.linalg.norm(w)**2 + self.C*loss