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

3- Implémenter un modèle de prédiction(simpliste) de machine learning: Random Forest et XGboost

4- Implémenter un modèle plus complexe de deep learning: RNN- LSTM

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


Tunning du paramètre de profondeur maximale
![max_depth](https://user-images.githubusercontent.com/44533474/163691697-a4ca4345-2e22-452c-a2a3-ec9c79c1bce3.png)

Tunning du paramètre du nombre d'estimateurs
![n_estimators](https://user-images.githubusercontent.com/44533474/163691732-8d5ade00-17f9-4eb8-92b2-2e7234aeda7c.png)


On fait les mêmes étapes pour XGboost et on obtient

|              | RandomForest(max_depth=none, n_estim=100) | RandomForest(max_depth=5, n_estim=10) | XGBoost(max_depth=4, n_estim=20) | XGBoost(max_depth=5, n_estim=20) |
|:------------------------:|:-:|:-:|:-:|:-:|
|        MSE Train     | 0.00107 |  0.003329 |  0.003307 | 0.003290  |
|        MSE Val       |  0.00749 |  0.00702 |  0.007016 | 0.00704 |






## Neural Network Model

Nous commençons cette section pareil comme pour les modèles random forest et xgboost. À savoir, nous prenons le même ensemble de données avec lequel nous
avons travaillé précédemment et nous nous occupons des données manquantes et de 'timestamp gaps'
 
  Nous allons utiliser un réseau récurrent de neurones (RNN)

![Architecture](https://user-images.githubusercontent.com/44533474/163689477-a7570a43-e69c-4510-8f62-2b802ae68327.png)
Référence d'images: https://keras.io/api/layers/recurrent_layers/

U,V,W sont des matrices de paramètres

X_t est l'entrée à l'instant t

h(t) sont les étas cachés:  h(t) = tanh (W* h(t-1)+ U(t))

y_t est la sortie à l'instant t 


Chaque neurone est assigné à un pas de temps fixé. La sortie de la couche cachée d'un pas de temps fait partie de l'entrée du pas de temps suivant.
L'algorithme consiste à trouver les matrices de poids optimales U,V,W qui donne la meilleure prédiction ou minimise la fonction de perte J.

Nous ferons plusieurs "train-test splits" donc on écrit une fonction: mysplit. On choisi 70% de données pour training, 20% pour validation et 10% pour test.
Ensuite on va normaliser les données.

LSTM: windowing

Les modèles font un ensemble de prédictions basées sur une fenêtre d'échantillons consécutifs à partir des données


Exemple : pour faire une seule prédiction 24 heures dans le futur, compte tenu de 24 heures d'historique, vous pouvez définir une fenêtre comme celle-ci :

![window](https://user-images.githubusercontent.com/44533474/163690113-661b944e-1e21-4f08-a9c8-90239d2fb001.png)


*width* (le nombre de pas de temps): largeur des fenêtres d'entrée et d'étiquette.
*shift* : décalage entre eux.


Les données utilisées pour le training du modèle sont au format tf.data.Dataset qui est divisé en entrées et étiquettes. De même pour les données d'évaluation et de test.

#### Design du modèle
Couche LSTM avec 20 unités internes

Couche dense à 2 unités. Les modèles denses sont traités à chaque pas de temps indépendamment.

Les unités sont choisies par validation croisée.


Les réseaux LSTM sont un type de réseau RNN capable d'apprendre la dépendance d'ordre dans les problèmes de prédiction de séquence



La photo ci-dessous montre ce que sont la couche et l'unité (ou neurone), et l'image la plus à droite montre la structure interne d'une seule unité LSTM.

![lstm_det](https://user-images.githubusercontent.com/44533474/163690535-86f85f68-236f-432e-8d4c-227ee9cc1cd3.png)



### Indicateurs de performances
On regarde le loss, MAS et MSE

### Le tuning de hyperparamètres

On utilise la validation croisée K-fold avec un ensemble Holdout.

K-fold for time series needs rolling basis: sklearn.model_selection.TimeSeriesSplit.

Enfin on applique une régularisation L2.

### Faire un update pour LSTM avec de nouvelles données 
(ref: https://machinelearningmastery.com/update-neural-network-models-with-more-data/)

### Multi-step prediction

prédire toutes les caractéristiques sur tous les pas de temps de sortie.

Pour le multi-step model, les données d'apprentissage sont  constituées d'échantillons horaires. Ici, les modèles apprendront à prédire 15 pas dans le futur, étant donné 4 pas du passé.



|              |  RandomForest | XGBoost | LSTM | 
|:------------------------:|:-:|:-:|:-:|
|        MSE Train     | 0.003329 |  0.003290 |  0.0034 | 
|        MSE Val       |  0.00702|  0.00704 |  0.0035 | 

