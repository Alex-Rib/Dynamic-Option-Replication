#############################################################
# importation des bibliothèques nécessaires
############################################################# 

import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt

#############################################################
# fonctions Black-Scholes et delta
############################################################# 

def black_scholes_call_price(ST, K, r, sigma, tau):
    d1 = (np.log(ST/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    call_price = ST*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    return call_price


def black_scholes_delta(ST, K, r, sigma, tau):
    d1 = (np.log(ST/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    delta = norm.cdf(d1)
    return delta   

#############################################################
# fonctions delta-hedging
#############################################################

def simple_delta_hedging(call_prices, deltas, ST, r):

    n = len(ST)  
    dt = 1/252  

    # initialisation du portefeuille et du compte bancaire
    portfolio = np.zeros(n)
    bank_account = np.zeros(n)

    # initialisation au temps 
    bank_account[0] = call_prices[0] - deltas[0] * ST[0]  # on achète delta[0] actions
    portfolio[0] = deltas[0] * ST[0] + bank_account[0] # valeur initiale du portefeuille : portfolio[0] = call_prices[0]
    # boucle de rebalancement quotidien
    for i in range (1,n):
        interet = bank_account[i-1] * (np.exp(r*dt)-1) # intérêts sur le compte bancaire 
        bank_now = bank_account[i-1] + interet # BT = B_T-1 + B_T-1*exp(r*dt) - B_T-1
        # rebalancement du portefeuille
        
        if i < n - 1: # pas de rebalancement au dernier pas de temps

            delta_diff = (deltas[i] - deltas[i-1]) * ST[i] 
            bank_account[i] = bank_now - delta_diff
            portfolio[i] = bank_account[i] + deltas[i] * ST[i]

        else : # au dernier pas de temps, on liquide la position en actions

            bank_account[i] = bank_now
            portfolio[i] = bank_account[i] + deltas[i-1] * ST[i]

    return portfolio, bank_account

#############################################################
# fonctions pour calculer la volatilité 
#############################################################

def vol_logreturns(ST):
    log_returns = np.log(ST[1:] / ST[:-1])
    return np.std(log_returns) * np.sqrt(252)


def GK_vol(data):
    O = data['Open'].values
    H = data['High'].values
    L = data['Low'].values
    C = data['Close'].values
    
    daily_variance = ((0.5 * (np.log(H / L)) ** 2) - ((2 * np.log(2) - 1) * (np.log(C / O)) ** 2))
    sigma_gk = np.mean(daily_variance)
    vol = np.sqrt(sigma_gk * 252)
    
    return vol 

def main():
    #############################################################
    # les paramètres de l'option
    ############################################################# 

    data = yf.Ticker("NVDA").history(period="3mo", interval="1d")
    ST = np.array(data['Close'].values)
    # Calcul du temps à l'échéance pour chaque jour
    tau = np.array([(len(ST) - 1 - i)/252 for i in range(len(ST))])
    tau = np.maximum(tau, 1/252)  # éviter tau=0
    K = ST[0]  # prix d'exercice
    r = 0.04  # taux sans risque annuel
    sigma = 0.18  # volatilité annuelle

    ####################################################################################
    # calcul de la stratégie de delta-hedging avec différentes volatilités
    #################################################################################### 

    ### volatilité choisie 
    call_prices = black_scholes_call_price(ST, K, r, sigma, tau)
    deltas = black_scholes_delta(ST, K, r, sigma, tau)
    portfolio, bank_account = simple_delta_hedging(call_prices, deltas, ST, r)


    ### volatilité log-returns
    sigma_log = vol_logreturns(ST)
    call_prices_log = black_scholes_call_price(ST, K, r, sigma_log, tau)
    deltas_log = black_scholes_delta(ST, K, r, sigma_log, tau)
    portfolio_log, bank_account_log = simple_delta_hedging(call_prices_log, deltas_log, ST, r)


    ### volatilité de Garman-Klass
    sigma_gk = GK_vol(data)
    call_prices_gk = black_scholes_call_price(ST, K, r, sigma_gk, tau)
    deltas_gk = black_scholes_delta(ST, K, r, sigma_gk, tau)
    portfolio_gk, bank_account_gk = simple_delta_hedging(call_prices_gk, deltas_gk, ST, r)

    ###################################################################
    # affichage de l'évolution du prix de l'option selon la volatilité
    ###################################################################

    sigmas = np.array([sigma, sigma_gk, sigma_log])

    plt.figure(figsize=(12, 6))
    for s in sigmas:
        call_prices_s = black_scholes_call_price(ST, K, r, s, tau)
        plt.plot(call_prices_s, label=f" volatilité ={s:.4f}")

    plt.title("évolution du prix du Call selon la volatilité")
    plt.xlabel("Jours")
    plt.ylabel("Prix du Call")
    plt.legend() 
    plt.grid()
    plt.show()

    ###################################################################
    # affichage de l'évolution du delta de l'option selon la volatilité
    ###################################################################

    plt.figure(figsize=(12, 6))
    for s in sigmas:
        deltas_S = black_scholes_delta(ST, K, r, s, tau)
        plt.plot(deltas_S, label=f" volatilité ={s:.4f}")

    plt.title("évolution du delta selon la volatilité")
    plt.xlabel("Jours")
    plt.ylabel("Delta")
    plt.legend() 
    plt.grid()
    plt.show()

    ###################################################################
    # affichage de l'évolution du compte bancaire selon la volatilité
    ###################################################################

    plt.figure(figsize=(12, 6))
    plt.plot(bank_account, label=f" volatilité choisie ={sigma:.3f}")
    plt.plot(bank_account_log, label=f" volatilité log-returns ={sigma_log:.3f}")
    plt.plot(bank_account_gk, label=f" volatilité Garman-Klass ={sigma_gk:.3f}")
    plt.title("évolution du compte bancaire selon la volatilité")
    plt.xlabel("Jours")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.show()

    #######################################################################
    # affichage de la stratégie de delta-hedging pour la volatilité choisie
    #######################################################################

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio, label="Portefeuille")
    plt.plot(call_prices, label="Prix du Call")
    plt.title(label = f"Comparaison de notre stratégie de couverture avec volatilité choisie ={sigma:.3f} vs prix du Call")
    plt.xlabel("Jours")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.show()

    ###########################################################################
    # affichage de la stratégie de delta-hedging pour la volatilité log-returns
    ###########################################################################

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_log, label="Portefeuille")
    plt.plot(call_prices_log, label="Prix du Call")
    plt.title(label = f"Comparaison de notre stratégie de couverture avec volatilité log-returns ={sigma_log:.3f} vs prix du Call")
    plt.xlabel("Jours")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.show()

    ###########################################################################
    # affichage de la stratégie de delta-hedging pour la volatilité Garman-Klass
    ###########################################################################

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_gk, label="Portefeuille")
    plt.plot(call_prices_gk, label="Prix du Call")
    plt.title(label = f"Comparaison de notre stratégie de couverture avec volatilité Garman-Klass ={sigma_gk:.3f} vs prix du Call")
    plt.xlabel("Jours")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

