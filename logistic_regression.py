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
    def __init__(self, DTR, LTR, l, pt=None):
        self.DTR = DTR
        self.ZTR = LTR * 2.0 -1.0
        self.LTR = LTR
        self.l = l
        self.pt=pt

    def logreg_obj(self, v):
        w, b = vcol(v[0:-1]), v[-1]
        scores = np.dot(w.T, self.DTR) + b
        if(self.pt is not None):
            S0 = np.dot(w.T, self.DTR[:, self.LTR == 0]) + b
            S1 = np.dot(w.T, self.DTR[:, self.LTR == 1]) + b
            loss_per_sample = self.pt * np.logaddexp(0, -S1).mean()
            loss_per_sample += (1 - self.pt) * np.logaddexp(0, S0).mean()
            return loss_per_sample + 0.5 * self.l * np.linalg.norm(w) ** 2
        else:
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
        # if self.pt is not None:
        #     scores = scores - np.log(self.DTR[:, self.LTR == 1].shape[1] / self.DTR[:, self.LTR == 0].shape[1]  )
        return scores
    
    def test_logreg_cali(self, DTR, LTR, DTE):
        self.DTR  = DTR
        self.LTR = LTR
        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True)
        _w = _v[0:DTR.shape[0]]
        _b = _v[-1]
        calibration =  np.log(self.pt / (1 - self.pt))
        self.b = _b
        self.w = _w
        STE = np.dot(_w.T, DTE) + _b - calibration
        return _w, _b, calibration   