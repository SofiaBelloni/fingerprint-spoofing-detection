import sys
sys.path.append('c:\\Users\\ferla\\Documents\\GitHub\\MLPR-Fingerprint-Spoofing-Detection')
from logistic_regression import *
from project import *
from metrics import *
from GMM_classifier import GMM_classifier


def k_fold_cv(features_train, labels_train, k, n=2):
    #1. Dividere il training set in K fold
    dim = features_train.shape[1]
    max_elem = dim // k
    features_folds = []
    labels_folds = []

    GMM_full_tied_scores = []
    GMM_full_untied_scores = []
    GMM_naive_tied_scores = []
    GMM_naive_untied_scores = []

    znorm_GMM_full_tied_scores = []
    znorm_GMM_full_untied_scores = []
    znorm_GMM_naive_tied_scores = []
    znorm_GMM_naive_untied_scores = []

    labels = []
    for i in range(k):
        if i != k - 1:
            features_folds.append(features_train[:, i*max_elem : (i+1)*max_elem])
            labels_folds.append(labels_train[i*max_elem : (i+1)*max_elem])
        else:
            features_folds.append(features_train[:, i*max_elem::])
            labels_folds.append(labels_train[i*max_elem::])
    #Calcolo il dataset
    eval = []
    labels_eval = []
    for i in range(k):
        train = []
        labels_training = []
        for j in range(k):
            if (i == j):
                eval = features_folds[i] 
                labels_eval = labels_folds[i]
            else:
                train.append(features_folds[j])
                labels_training.append(labels_folds[j])
        train = np.hstack(train)
        labels_training = np.hstack(labels_training)
        znorm_train, znorm_eval = znorm(train, eval)
        GMM_model = GMM_classifier(train, labels_training, n)
        znorm_GMM_model = GMM_classifier(znorm_train, labels_training, n)
        ### RAW FEATURES###
        score_full_tied = GMM_model.compute_scores(eval, 'full', 'tied')    
        score_naive_tied = GMM_model.compute_scores(eval, 'naive', 'tied')    
        score_full_untied = GMM_model.compute_scores(eval, 'full', 'untied')    
        score_naive_untied = GMM_model.compute_scores(eval, 'naive', 'untied')
        ### ZNORM FEATURES ###
        znorm_score_full_tied = znorm_GMM_model.compute_scores(znorm_eval, 'full', 'tied')    
        znorm_score_naive_tied = znorm_GMM_model.compute_scores(znorm_eval, 'naive', 'tied')    
        znorm_score_full_untied = znorm_GMM_model.compute_scores(znorm_eval, 'full', 'untied')    
        znorm_score_naive_untied = znorm_GMM_model.compute_scores(znorm_eval, 'naive', 'untied')

        ### RAW FEATURES ###
        GMM_full_tied_scores.append(score_full_tied)
        GMM_full_untied_scores.append(score_full_untied)
        GMM_naive_tied_scores.append(score_naive_tied)
        GMM_naive_untied_scores.append(score_naive_untied)
        ### ZNORM FEATURES ###
        znorm_GMM_full_tied_scores.append(znorm_score_full_tied)
        znorm_GMM_full_untied_scores.append(znorm_score_full_untied)
        znorm_GMM_naive_tied_scores.append(znorm_score_naive_tied)
        znorm_GMM_naive_untied_scores.append(znorm_score_naive_untied)

        labels.append(labels_eval)
    ### RAW FEATURES ###
    GMM_full_tied_scores = np.hstack(GMM_full_tied_scores)
    GMM_full_untied_scores = np.hstack(GMM_full_untied_scores)
    GMM_naive_tied_scores = np.hstack(GMM_naive_tied_scores)
    GMM_naive_untied_scores = np.hstack(GMM_naive_untied_scores)
    ### ZNORM FEATURES ###
    znorm_GMM_full_tied_scores = np.hstack(znorm_GMM_full_tied_scores)
    znorm_GMM_full_untied_scores = np.hstack(znorm_GMM_full_untied_scores)
    znorm_GMM_naive_tied_scores = np.hstack(znorm_GMM_naive_tied_scores)
    znorm_GMM_naive_untied_scores = np.hstack(znorm_GMM_naive_untied_scores)
    labels = np.hstack(labels)
        
    ### RAW FEATURES ###
    minDCF_full_tied = compute_minDCF(pi, Cfn, Cfp, GMM_full_tied_scores, labels)
    minDCF_full_untied = compute_minDCF(pi, Cfn, Cfp, GMM_full_untied_scores, labels)
    minDCF_naive_tied = compute_minDCF(pi, Cfn, Cfp, GMM_naive_tied_scores, labels)
    minDCF_naive_untied = compute_minDCF(pi, Cfn, Cfp, GMM_naive_untied_scores, labels)
    ### ZNORM FEATURES ###
    znorm_minDCF_full_tied = compute_minDCF(pi, Cfn, Cfp, znorm_GMM_full_tied_scores, labels)
    znorm_minDCF_full_untied = compute_minDCF(pi, Cfn, Cfp, znorm_GMM_full_untied_scores, labels)
    znorm_minDCF_naive_tied = compute_minDCF(pi, Cfn, Cfp, znorm_GMM_naive_tied_scores, labels)
    znorm_minDCF_naive_untied = compute_minDCF(pi, Cfn, Cfp, znorm_GMM_naive_untied_scores, labels)

    return minDCF_full_tied, minDCF_full_untied, minDCF_naive_tied, minDCF_naive_untied, \
            znorm_minDCF_full_tied, znorm_minDCF_full_untied, znorm_minDCF_naive_tied, znorm_minDCF_naive_untied
            


def plot_hist(n_dim, minDCF, minDCF_norm, name):
    plt.figure()
    plt.bar(range(n_dim) - 0.2, minDCF, 0.35, label="RAW")
    plt.bar(range(n_dim) + 0.2, minDCF_norm, 0.35, label="z_norm")
    plt.legend()
    plt.tight_layout()
    plt.xlabel('GMM Components')
    plt.ylabel('minDCF')
    plt.xticks(range(n_dim), n_dim ** 2)
    plt.savefig('.\GMM\%s.png' % (name) )


features_train, labels_train = load_dataset('Train.txt')
features_train, labels_train = shuffle_dataset(features_train, labels_train)


full_tied = []
full_untied = []
naive_tied = []
naive_untied = []
znorm_full_tied = []
znorm_full_untied = []
znorm_naive_tied = []
znorm_naive_untied = []

minDCF_full_tied, minDCF_full_untied, minDCF_naive_tied, minDCF_naive_untied, \
            znorm_minDCF_full_tied, znorm_minDCF_full_untied, znorm_minDCF_naive_tied, znorm_minDCF_naive_untied = k_fold_cv(features_train, labels_train, 10, 4) 


# N = [1, 2, 3, 4, 5]

# for n in N:
#     minDCF_full_tied, minDCF_full_untied, minDCF_naive_tied, minDCF_naive_untied, \
#             znorm_minDCF_full_tied, znorm_minDCF_full_untied, znorm_minDCF_naive_tied, znorm_minDCF_naive_untied = k_fold_cv(features_train, labels_train, 10, n)    
#     print(f'Ho fatto la k fold - n = {n}')
#     full_tied.append(minDCF_full_tied)
#     full_untied.append(minDCF_full_untied)
#     naive_tied.append(minDCF_naive_tied)
#     naive_untied.append(minDCF_naive_untied)
#     znorm_full_tied.append(znorm_minDCF_full_tied)
#     znorm_full_untied.append(znorm_minDCF_full_untied)
#     znorm_naive_tied.append(znorm_minDCF_naive_tied)
#     znorm_naive_untied.append(znorm_minDCF_naive_untied)

# plot_hist(N, full_tied, znorm_full_tied, "full_tied")
# plot_hist(N, full_untied, znorm_full_untied, "full_untied")
# plot_hist(N, naive_tied, znorm_naive_tied, "naive_tied")
# plot_hist(N, naive_untied, znorm_naive_untied, "naive_untied")
