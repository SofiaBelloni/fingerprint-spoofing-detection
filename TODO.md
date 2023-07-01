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

## LDA

## Altro