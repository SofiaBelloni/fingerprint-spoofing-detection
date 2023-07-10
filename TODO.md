# TODO List 
## Introduction
**Qual è la task**
Classificazione binaria per individuare quando un fingerprint è autentico oppure falso.

**Che cos'è il dataset**
Il dataset consiste in dati sintetici per motivi di privacy, e le dimensioni delle features sono nettamente diminuiti dai casi d'uso reali.
The embeddings are 10-dimensional, continuous-valued vectors, belonging to either the authentic.
Tali componenti non hanno una interpretazione fisica.

**Dimensioni del dataset**
2325 immagini di train
Quanti di questi sono 0? 1525
Quanti di questi sono 1? 800
7704 immagini di test
Quanti di questi sono 0? 5304
Quanti di questi sono 1? 2400

Trovare una soluzione allo sbilanciamento delle classi.
Possibile idea: reduction con minimax facility location

**Cos'è questa cosa non lo so poi lo vedo ma intanto la metto che è utile (forse)**
The target application working point is threrefore defined by the triplet
(PiT = 0:5;Cfn = 1;Cfp = 10).

## Caricare il dataset
Ok fatto
## Rappresentare il dataset
**Matrice 10x10, da capire meglio che cosa stiamo plottando**
Sulla diagonale gli istogrammi, per tutto il resto gli scatter plot
**Effettuare degli scatter plot**
Ok fatto
**Istogrammi**
Ok fatto
## PCA
**Scrivere algoritmo per calcolo della PCA**
Ok fatto
**Trovare il valore di M ottimo**
Tale valore lo possiamo ottenere in due modi:
1. Considerandolo come iperparametro e quindi usare un validation set per stimarlo
2. Usare la formula per calcolare il valore ottimo, ma dobbiamo definire la soglia t (che nelle slides ad esempio è 95% = 0.95)

**Intanto, per plottare la PCA, definiamo m=2.** 
Ok fatto

Per poi valutare effettivamente qual è il contributo di tale tecnica, dobbiamo fare dei test con valori di m diversi, e vedere in che modo questi contribuiscono ad una migliore precisione. (Per ogni metodo di classificazione che implementeremo).

08.07 - 
* Plottare la varianza della PCA rispetto alle sue dimensioni.
Da questo, prendere le top 3 dimensioni della PCA con la percentuale di varianza più alta.
* Calcolare minDCF
* * Cos'è minDCF? Un modo per valutare le performance dei nostri modelli (molto generale)

## LDA
fatta 

## Pearson correlation
Fatto, da analizzare

## K fold cross validation
* Dividere il training set in K parti.
* Utilizzare K - 1 parti come training set e la parte rimanente come validation set (che è un test set "ridotto" dal quale valutiamo le performance per quell'iperparametro selezionato)
Ok fatto


## MVG Classifiers
### 1. Standard version
Ok fatto
Ricordarci (eventualmente) di inserire anche la versione logaritmica che dona stabilità numerica. In questo momento non è fatta così.

### 2. Naive Bayes
Utilizzo della matrice delle covarianze usata per il punto 1, moltiplicata per la matrice identità.
Ok fatto.

### 3. Tied Classifier
Ok fatto.

## Logistic Regression
ok fatto
## quadratic features expansion 
È importante notare che, sebbene la quadratic feature expansion possa essere efficace nel catturare modelli non lineari, può anche portare a uno spazio di feature più ampio e aumentare potenzialmente il rischio di overfitting. Pertanto, è importante bilanciare attentamente la complessità dell'espansione delle feature con i dati disponibili e le prestazioni del modello. Le tecniche di regolarizzazione, come la regolarizzazione L1 o L2, possono aiutare a mitigare l'overfitting nei modelli di regressione logistica.

## Support Vector Machines
C è un iperparametro da testare con valori 0.1, 1, 10
Ok fatto

## Gaussian Mixture Models

## Calibration and fusion

## Evaluation

## Scrittura del report



