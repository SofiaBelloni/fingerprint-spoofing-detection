from project import vcol
import numpy as np
import scipy

def quadratic_features_expansion(x, l):
    # Aggiungi le feature espandenti quadratiche
    x_expanded = np.hstack((x, x ** 2))

    # Aggiungi un termine di bias alla matrice delle feature
    l_expanded = np.hstack((l,l))
    return x_expanded, l_expanded

class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.ZTR = LTR * 2.0 -1.0
        self.l = l

    def logreg_obj(self, v):
        w, b = vcol(v[0:-1]), v[-1]
        scores = np.dot(w.T, self.DTR) + b
        loss_per_sample = np.logaddexp(0, -self.ZTR * scores)
        loss = loss_per_sample.mean() + 0.5 * self.l * np.linalg.norm(w)**2
        return loss
    
    def train(self):
        x0 = np.zeros(self.DTR.shape[0]+1) #create initial value
        # x is the estimated position of the minimum
        # f is the objective value at the minimum
        # d contains additional information (check the documentation)
        xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0,  approx_grad = True)
        return xOpt
    
    def compute_scores(self, DTE):
        xOpt = self.train()
        w, b = xOpt[0:-1], xOpt[-1]
        scores = np.zeros(DTE.shape[1])
        for i in range(DTE.shape[1]):
            scores[i] = np.dot(np.transpose(w), DTE[:, i]) + b
        return scores
