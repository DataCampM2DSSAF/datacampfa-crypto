# datacampfa-crypto
datacampfa-crypto created by GitHub Classroom

# Lundi 10 Janvier

Dans la première séance on a géré la partie logistique du projet: explorer les variables, voir la base de données et bien comprendre le but de ce challenge. 
On a aussi créé nos comptes sur Kaggle et github, ainsi les groupes sur chacun d'eux.

# Lundi 24 janvier
On a enchaîné cette semaine avec la compréhension du sujet, puis nous avons essayé de traiter les valeurs manquantes.

De plus nous avons mis un plan de travail pour avancer dans les séances prochaines :

1- Description de la base de données (relations entre les variables, corrélation..)

2- Étudier et traiter les valeurs manquantes

3- Implémenter un modèle de prédiction: GARCH

#Iota est décorrélé de tous les autres devises

# Mardi  1 février 

En utilisant le repisotory https://github.com/manthanthakker/BitcoinPrediction pour s'inspirer,il présente des implémentations d'algorithmes d'apprentissage automatique (Random Forest, régression, etc.) et de réseaux neuronaux récurrents / réseaux à mémoire à long terme pour la prédiction de BitCoin. De plus, dans notre cas, nous avons identifié que BitCoin est la monnaie la plus importante, car la plupart des autres monnaies numériques suivront de près ses tendances. Ainsi, disposer d'un modèle de prédiction précis du BitCoin devrait être une partie essentielle du projet.


# Lundi 14 Février 
Nous avons commencé cette séance par comprendre le diagramme de GANTT pour la gestion de notre projet et puis après avoir importer les données obtenues à partir de Kaggle de notre base de données qui contient des informations historiques de plusieurs cryptomonnaies comme Bitcoin et Ethereum, on a passé à l’étape de préparation et nettoyage des données.


# Lundi 21 Février 
La tâche d'aujourd'hui c'est de traiter les valeurs manquantes, c'est plutôt dans la différence d'intervalle des timestamps . 
Nous avons commencé par extraire chaque crytomonnaie avec les timestamps correspondants, les visualiser pour mieux détecter les différences et puis imputer chaque valeurs manquantes par la moyenne de la valeur d'avant et la valeur d'après.



# 10 - 22 Mars


## Description des données

Le training set a les variables suivantes:

1- timestamp - A timestamp for the minute covered by the row.

2- Asset_ID - ID code pour chaque cryptomonnaie

3- Count - Le nombre de transactions qui ont eu lieu cette minute.

4- Open - Le prix en USD au début de la minute.

5- High - Le prix le plus élevé en USD pendant la minute.

6- Low -  Le prix le plus bas en USD pendant la minute.

7- Close - Le prix en USD à la fin de la minute.

8- Volume -Le nombre d'unités de crypto-monnaies échangées pendant la minute.

9- VWAP - Le prix moyen pondéré en fonction du volume pour la minute.

10- Target - Rendements résiduels de 15 minutes.


## Données manquantes

On a sélectionné une partie des données comme un 'working batch'. 

La variable timestamp indique l'heure à laquelle toutes les variables ont été enregistrées. Tout d'abord, nous prenons une partie des données, examinons un actif individuel et convertissons l'horodatage en dates lisibles par l'humain.  Les données présentent des mises à jour des valeurs pour chaque minute, mais des valeurs manquantes apparaissent et nous devons résoudre ce problème. Nous résolvons cela localement, en utilisant la méthode panda 'reindex' pour chaque 'Asset_ID';  chaque intervalle de temps manquant est rempli avec le dernier échantillon pertinent. Nous créons les comuns 'heure' et 'jour'.


Tout d'abord, nous examinons les variables avec des valeurs 'target'(10) manquantes. Il s'agit de moins de 2 pourcent pour ce genre de données manquantes et nous décidons de les éliminer. En effet, des données manquantes apparaissent et nous devons résoudre cela.

### Feature Engineering 

Nous créons les variables 'hour' et 'day'. On normalise les variables numériques Count, Open, High, Low, Close, Volume et VWAP de 0 à 1.

### Visualisation

On a fait des 'time series' plots et une matrice de corrélation pour voir la relation entre chaque variable et la variable 'target'.

### Prediction

On a entraîné un modèle RandomForestRegressor avec une partie des données. Les résultats sur les données de test sélectionnées étaient satisfaisants.



Nous avons ajouté un modèle de gradient boost. De plus, nous avons ajouté une fonction de "hyperparameter tunning" pour nos modèles et nous avons fait des graphiques de RMSE.

La prochaine étape que nous choisissons consiste à prédire les devises à l'aide de réseaux de neurones. Nous pensons utiliser un RNN LSTM comme nous l'avons vu utilisé dans la prédiction BitCoin.


## Neural Network Model

Nous commençons cette section à l'identique par celle des modèles random forest et xgboost. À savoir, nous prenons le même ensemble de données avec lequel nous
avons travaillé précédemment et nous nous occupons des données manquantes et de 'timestamp gaps'
 
  Nous allons utiliser un réseau récurrent de neurones (RNN)

![Architecture](https://user-images.githubusercontent.com/44533474/163689477-a7570a43-e69c-4510-8f62-2b802ae68327.png)

U,V,W sont des matrices de paramètres

X_t est l'entrée à l'instant t

h(t) sont les étas cachés:  h(t) = tanh (W* h(t-1)+ U(t))

y_t est la sortie à l'instant t 


Chaque neurone est assigné à un pas de temps fixé. La sortie de la couche cachée d'un pas de temps fait partie de l'entrée du pas de temps suivant.
L'algorithme consiste à trouver les matrices de poids optimales U,V,W qui donne la meilleure prédiction ou minimise la fonction de perte J.
