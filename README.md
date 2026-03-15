# Réplication Dynamique d'Options par Delta-Hedging

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Derivatives-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

## 📋 Description

Ce projet implémente une stratégie de **réplication dynamique d'options** basée sur la méthode du **Delta-Hedging** .

L'objectif est de synthétiser le payoff d'un Call Européen en construisant un portefeuille auto-financé composé d'actifs risqués et d'actifs sans risque , rééquilibré quotidiennement selon les sensibilités du modèle de Black-Scholes.

> **Note Méthodologique :** Dans le cadre de projet pédagogique, certaines courbes utilisent la volatilité réalisée comme input. Ce choix permet d'isoler l'erreur de couverture due à la discrétisation temporelle en neutralisant le bruit lié à l'estimation de la volatilité .

## 🎯 Objectifs

- **Démontrer la réplication :** Prouver qu'un portefeuille $\Delta S + B$ converge vers le prix de l'option.
- **Gestion de Trésorerie :** Visualiser les flux de financement (Leverage) nécessaires au maintien de la position.
- **Analyse de Volatilité :** Comparer l'impact de différents estimateurs sur la couverture.

## 📊 Méthode 

### 1. Modèle de Black-Scholes
Le prix du Call et son Delta ($\Delta$) sont calculés selon les formules fermées :

$$C(S_t, t) = S_t \Phi(d_1) - K e^{-r\tau} \Phi(d_2)$$

$$\Delta = \frac{\partial C}{\partial S} = \Phi(d_1)$$

Où :
- $d_1 = \frac{\ln(S_t/K) + (r + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$
- $d_2 = d_1 - \sigma\sqrt{\tau}$

### 2. Algorithme de Réplication
À chaque pas de temps $t$, nous ajustons le portefeuille pour qu'il soit **Delta-Neutre** par rapport à l'option vendue :

1.  Calcul du nouveau $\Delta_t$.
2.  Achat/Vente de $( \Delta_t - \Delta_{t-1} )$ actions au prix marché $S_t$.
3.  Le coût de l'opération est financé par le compte bancaire $B_t$ (qui accumule des intérêts au taux $r$).

L'équation du portefeuille est : $V_t = \Delta_t S_t + B_t$

## 🔧 Fonctionnalités

### Trois estimateurs de volatilité

1. **Volatilité fixe** : volatilité choisie manuellement (σ = 18%)
2. **Volatilité des log-returns** : calculée à partir des rendements logarithmiques historiques
3. **Volatilité de Garman-Klass** : estimateur utilisant les prix Open/High/Low/Close

### Visualisations

- Évolution du prix du call selon la volatilité
- Évolution du delta selon la volatilité
- Évolution du compte bancaire
- Comparaison portefeuille de réplication vs prix théorique du call (pour chaque volatilité)

## 📦 Dépendances

```python
numpy
scipy
yfinance
matplotlib
```

Installation :
```bash
pip install numpy scipy yfinance matplotlib
```

## 🚀 Utilisation

Le script utilise des données réelles de NVIDIA (NVDA) sur les 3 derniers mois :

```python
python hedging.py
```

Le programme génère automatiquement 6 graphiques comparant les performances de la stratégie de delta-hedging avec les trois paramètres de volatilité.

## 📈 Paramètres

Les paramètres principaux peuvent être modifiés dans le script :

- `Ticker` : symbole de l'action (défaut : "NVDA")
- `period` : période historique (défaut : "3mo")
- `K` : prix d'exercice (défaut : prix initial $S_0$)
- `r` : taux sans risque annuel (défaut : 4%)
- `sigma` : volatilité annuelle choisie (défaut : 18%)

## 📊 Résultats

Le code permet de visualiser :
- L'efficacité de la réplication dynamique
- L'impact du choix de la volatilité sur la performance de la couverture
- L'évolution du compte bancaire nécessaire pour maintenir la couverture

## 🔬 Concepts Théoriques

### Delta-Hedging
Stratégie de couverture visant à neutraliser le risque lié aux variations du prix du sous-jacent en ajustant dynamiquement la position en actions.

### Réplication Dynamique
Construction d'un portefeuille auto-finançant qui réplique le payoff d'un produit dérivé sans détenir directement ce dérivé.

### Volatilité fixée vs Historique
Comparaison entre une volatilité fixée a priori et des estimateurs basés sur les données historiques (log-returns, Garman-Klass).

## 📝 Notes

- Le rééquilibrage est effectué quotidiennement (dt = 1/252)
- Les intérêts sur le compte bancaire sont calculés en continu
- Au dernier pas de temps, la position en actions est liquidée

## 🎓 Applications

Ce projet est uniquement à but pédagogique et permet de :
- Comprendre les mécanismes de couverture en finance de marché
- Illustrer la théorie de Black-Scholes en pratique
- Analyser l'impact de la volatilité sur les stratégies de trading


## 👨‍💻 Auteur

Alexandre R. - Université Paris Cité
