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
Donc nous avons dans un premier temps un fichier train.csv contenant ces variables:
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
