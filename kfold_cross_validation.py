from project import *
import MVG_classifiers
import logistic_regression

def k_fold_cv(features_train, labels_train, k):
    #1. Dividere il training set in K fold
    dim = features_train.shape[1]
    max_elem = dim // k
    features_folds = []
    labels_folds = []
    MVG = []
    MVG_NB = []
    MVG_TIED = []
    LOG_REG = []
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
                labels_eval = labels_train[i]
            else:
                train.append(features_folds[j])
                labels_training.append(labels_folds[j])
        train = np.hstack(train)
        labels_training = np.hstack(labels_training)
        print('STAMPA RISULTATI')
        print(f'Training shape: {train.shape}')
        print(f'Evaluation shape: {eval.shape}')
        # Qui noi alleniamo
        # Chiamare i vari metodi di classificazione
        ### MVG ###
        labels_MVG = MVG_classifiers.gaussian_classifier(train, labels_training, eval)
        MVG.append(np.sum(labels_MVG == labels_eval) / labels_MVG.shape[0])
        print(f'MVG {i}: {MVG[i]}')
        ### NAIVE BAYES MVG ###
        labels_MVG_nb = MVG_classifiers.nb_gaussian_classifier(train, labels_training, eval)
        MVG_NB.append(np.sum(labels_MVG_nb == labels_eval) / labels_MVG_nb.shape[0])
        print(f'MVG NB {i}: {MVG_NB[i]}')
        ### TIED MVG ###
        labels_MVG_tied = MVG_classifiers.tied_classifier(train, labels_training, eval)
        MVG_TIED.append(np.sum(labels_MVG_tied == labels_eval) / labels_MVG_tied.shape[0])
        print(f'MVG TIED {i}: {MVG_TIED[i]}')
        ### LOGISTIC REGRESSION ###
        logRegMethod = logistic_regression.logRegClass(train, labels_training, 0.001)
        labels_LOG_REG = logRegMethod.predict(eval)
        LOG_REG.append(np.sum(labels_LOG_REG == labels_eval) / labels_LOG_REG.shape[0])
        print(f'LOGISTIC REGRESSION {i}: {LOG_REG[i]}')
    
    return np.array(MVG).mean(), np.array(MVG_NB).mean(), np.array(MVG_TIED).mean(), np.array(LOG_REG).mean()


        


features_train, labels_train = load_dataset('Train.txt')
features_test, labels_test = load_dataset('Test.txt')
MVG, MVG_NB, MVG_TIED, LOG_REG = k_fold_cv(features_train, labels_train, 10)

print(f'MVG mean accuracy {MVG}')
print(f'MVG NB mean accuracy {MVG_NB}')
print(f'MVG TIED mean accuracy {MVG_TIED}')
print(f'LOGISTIC REGRESSION mean accuracy {LOG_REG}')

#TODO: rispondere alla domanda come mai qui si comporta meglio tied MVG ma nel main ho risulatati nettamente migliori con LOG REG?
#TODO: ripetere kfold per tutti i metodi di classificazione
