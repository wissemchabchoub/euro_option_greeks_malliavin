import random
import warnings

import numpy as np
import tqdm
from scipy.integrate import dblquad, quad

warnings.filterwarnings('ignore')


def euro_call_price(x, n, r, sigma, k, T, N):
    """
    The euro_call_price function computes the price of a European call option using Monte Carlo simulation.

    Parameters
    ----------
        x
            Determine the number of steps in the binomial tree
        n
            Determine the number of time steps used in the binomial tree
        r
            Calculate the discount factor
        sigma
            Determine the volatility of the stock price
        k
            Define the strike price of the option
        T
            Calculate the maturity of the option
        N
            Determine the number of scenarios to generate

    Returns
    -------

        The price of a european call option, the empirical variance of the payoffs, the empirical standard deviation and 95% confidence interval

    """
    X_T = []
    for scenario in range(N):
        X_T.append(bs_price(x, n, r, sigma)[1, -1])
    X_T = np.array(X_T)
    payoffs = np.maximum(X_T-k, 0)
    price = np.mean(np.exp(-r*T)*payoffs)
    emp_variance = np.var(np.exp(-r*T)*payoffs)
    emp_std = np.std(np.exp(-r*T)*payoffs)
    CI_1 = price-1.96*emp_std/np.sqrt(N)
    CI_2 = price+1.96*emp_std/np.sqrt(N)
    return price, emp_variance, emp_std, CI_1, CI_2


def delta_gamma_euro_call_1(x, n, r, sigma, k, T, N, h):
    """
    The delta_gamma_euro_call_1 function computes the delta and gamma of a European call option using the finite difference method.

    Parameters
    ----------
        x
            Define the initial stock price
        n
            Determine the number of time steps
        r
            Denote the risk-free interest rate
        sigma
            Determine the volatility of the stock price
        k
            Determine the strike price of the option
        T
            Define the time horizon of the option
        N
            Determine the number of scenarios to generate
        h
            The diff term

    Returns
    -------

        Delta of a european call option, the empirical variance, the empirical standard deviation and 95% confidence interval
        Gamma of a european call option, the empirical variance, the empirical standard deviation and 95% confidence interval
    """

    X_T_h1 = []
    X_T_h2 = []
    X_T = []
    for scenario in range(N):
        scen = bs_price_three_paths(x, n, r, sigma, h)
        X_T_h1.append(scen[0, -1])
        X_T_h2.append(scen[1, -1])
        X_T.append(scen[2, -1])
    X_T_h1 = np.array(X_T_h1)
    X_T_h2 = np.array(X_T_h2)
    X_T = np.array(X_T)
    F_h1 = np.exp(-r*T)*np.maximum(X_T_h1-k, 0)
    F_h2 = np.exp(-r*T)*np.maximum(X_T_h2-k, 0)
    F = np.exp(-r*T)*np.maximum(X_T-k, 0)
    Gamma = np.mean((F_h1+F_h2-2*F)/(h*h))
    emp_variance_gamma = np.var((F_h1+F_h2-2*F)/(h*h))
    emp_std_gamma = np.std((F_h1+F_h2-2*F)/(h*h))
    CI_1_gamma = Gamma-1.96*emp_std_gamma/np.sqrt(N)
    CI_2_gamma = Gamma+1.96*emp_std_gamma/np.sqrt(N)
    Delta = np.mean((F_h2-F_h1)/(2*h))
    emp_variance_delta = np.var((F_h2-F_h1)/(2*h))
    emp_std_delta = np.std((F_h2-F_h1)/(2*h))
    CI_1_delta = Delta-1.96*emp_std_delta/np.sqrt(N)
    CI_2_delta = Delta+1.96*emp_std_delta/np.sqrt(N)
    return (Delta, emp_variance_delta, emp_std_delta, CI_1_delta, CI_2_delta), (Gamma, emp_variance_gamma, emp_std_gamma, CI_1_gamma, CI_2_gamma)


def delta_gamma_euro_call_2(x, n, r, sigma, k, T, N):
    """
    The delta_gamma_euro_call_2 function computes the delta and gamma of a European call option using Monte Carlo simulation and Malliavin Calculus.

    Parameters
    ----------
        x
            Represent the underlying asset price
        n
            Determine the number of time steps
        r
            Calculate the discount factor
        sigma
            Determine the volatility of the stock price
        k
            Define the strike price of the option
        T
            Calculate the maturity of the option
        N
            Determine the number of scenarios to be simulated in the monte carlo simulation

    Returns
    -------

        Delta of a european call option, the empirical variance, the empirical standard deviation and 95% confidence interval
        Gamma of a european call option, the empirical variance, the empirical standard deviation and 95% confidence interval

    """
    X_T = []
    B_T = []
    for scenario in range(N):
        scen = bs_price(x, n, r, sigma)
        X_T.append(scen[1, -1])
        B_T.append(scen[0, -1])
    X_T = np.array(X_T)
    B_T = np.array(B_T)
    F = np.maximum(X_T-k, 0)
    Delta = np.mean(np.exp(-r*T)*B_T*F/(x*sigma*T))
    emp_std_delta = np.std(np.exp(-r*T)*B_T*F/(x*sigma*T))
    emp_var_delta = np.var(np.exp(-r*T)*B_T*F/(x*sigma*T))
    CI_1_delta = Delta-1.96*emp_std_delta/np.sqrt(N)
    CI_2_delta = Delta+1.96*emp_std_delta/np.sqrt(N)
    Gamma = np.mean(np.exp(-r*T) * (-B_T/(x*x*sigma*T) +
                    (B_T**2-T)/(sigma*T*x)**2) * F)
    emp_std_gamma = np.std(np.exp(-r*T) * (-B_T/(x*x*sigma*T) +
                                           (B_T**2-T)/(sigma*T*x)**2) * F)
    emp_var_gamma = np.var(np.exp(-r*T) * (-B_T/(x*x*sigma*T) +
                                           (B_T**2-T)/(sigma*T*x)**2) * F)
    CI_1_gamma = Gamma-1.96*emp_std_gamma/np.sqrt(N)
    CI_2_gamma = Gamma+1.96*emp_std_gamma/np.sqrt(N)
    return (Delta, emp_var_delta, emp_std_delta, CI_1_delta, CI_2_delta), (Gamma, emp_var_gamma, emp_std_gamma, CI_1_gamma, CI_2_gamma)


def delta_euro_call(x, n, r, sigma, k, T, N, delta):
    """
    The delta_euro_call function computes the delta of a European call option.

    Parameters
    ----------
        x
            Represent the underlying asset price
        n
            Determine the number of time steps
        r
            Calculate the discount factor
        sigma
            Determine the volatility of the stock price
        k
            Define the strike price of the option
        T
            Calculate the maturity of the option
        N
            Determine the number of scenarios to be simulated in the monte carlo simulation
        delta
            delta term


    Returns
    -------

        The delta of the european call option with respect to the stock price, its empirical variance and its standard deviation, as well as a 95% confidence interval

    """
    H_vectorized = np.vectorize(H)
    F_vectorized = np.vectorize(F)
    X_T = []
    B_T = []
    for scenario in tqdm.tqdm(range(N)):
        scen = bs_price(x, n, r, sigma)
        X_T.append(scen[1, -1])
        B_T.append(scen[0, -1])
    X_T = np.array(X_T)
    B_T = np.array(B_T)
    final_values = np.exp(-r*T)*H_vectorized(X_T, k, delta)*X_T/x + \
        np.exp(-r*T)*(B_T*F_vectorized(X_T, k, delta))/(X_T*sigma*T)
    Delta = np.mean(final_values)
    emp_variance = np.var(final_values)
    emp_std = np.std(final_values)
    CI_1 = Delta-1.96*emp_std/np.sqrt(N)
    CI_2 = Delta+1.96*emp_std/np.sqrt(N)
    return Delta, emp_variance, emp_std, CI_1, CI_2


def gamma_euro_call(x, n, r, sigma, k, T, N, delta):
    """
    The delta_euro_call function computes the delta of a European call option.

    Parameters
    ----------
        x
            Represent the underlying asset price
        n
            Determine the number of time steps
        r
            Calculate the discount factor
        sigma
            Determine the volatility of the stock price
        k
            Define the strike price of the option
        T
            Calculate the maturity of the option
        N
            Determine the number of scenarios to be simulated in the monte carlo simulation
        delta
            delta term


    Returns
    -------

        The delta of the european call option with respect to the stock price, its empirical variance and its standard deviation, as well as a 95% confidence interval

    """
    I_vectorized = np.vectorize(I_)
    F_vectorized = np.vectorize(F_)
    X_T = []
    B_T = []
    for scenario in tqdm.tqdm(range(N)):
        scen = bs_price(x, n, r, sigma)
        X_T.append(scen[1, -1])
        B_T.append(scen[0, -1])
    X_T = np.array(X_T)
    B_T = np.array(B_T)
    final_values = np.exp(-r*T)*I_vectorized(X_T, 0, delta)*X_T*X_T/(x**2)+np.exp(-r*T) * \
        ((-B_T/(x**2*sigma*T) + (B_T*B_T-T)/((sigma*T*x)**2))
         * F_vectorized(X_T, k, delta))
    Gamma = np.mean(final_values)
    emp_variance = np.var(final_values)
    emp_std = np.std(final_values)
    CI_1 = Gamma-1.96*emp_std/np.sqrt(N)
    CI_2 = Gamma+1.96*emp_std/np.sqrt(N)
    return Gamma, emp_variance, emp_std, CI_1, CI_2

####### Helpers #######


def bs_price(x, n, r, sigma):
    """
    The bs_price function generates a path using the Black-Scholes formula.
    The function takes as input:
        x - The current stock price;
        n - The number of time steps to simulate; and, 
        r - The risk free interest rate. 

    Parameters
    ----------
        x
            Define the initial value of the stock price
        n
            Determine the number of steps in the brownian motion
        r
            Calculate the drift term
        sigma
            Calculate the standard deviation of the normal distribution

    Returns
    -------
        An array of two arrays (price and W)
    """
    X = [x]
    B = [0]
    W = 0
    for i in range(n):
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        dW = np.sqrt(-2*np.log(u)*(1.0)/float(n))*np.cos(2*np.pi*v)
        W = W+dW
        X.append(x*np.exp(sigma*W-(1/2*sigma**2-r)*(i+1)/n))
        B.append(W)
    return np.array([B, X])


def bs_price_three_paths(x, n, r, sigma, h):
    """
    The bs_price_three_paths function generates a path using the Black-Scholes formula.
    p(x + h) , p(x) p(x − h)
    The function takes as inputs:
        x - The current value of the underlying asset, which is being modeled as a stock.
        n - The number of time steps to compute for each path.  This is equivalent to the number of binomial iterations we will run our algorithm for.  In other words, if we want to model an option with one year remaining until expiration, then n = 252 (252 trading days in a year).
        r - The risk free interest rate per annum that will be used in computing
        h - The diff term 

    Parameters
    ----------
        x
            Set the initial value of the asset price
        n
            Determine the number of time steps
        r
            Calculate the drift of the stock price
        sigma
            Calculate the standard deviation of the normal distribution used to generate random numbers
        h
            The diff term

    Returns
    -------

        A 3-dimensional array

    """
    X1 = [x-h]
    X2 = [x+h]
    X3 = [x]
    W = 0
    for i in range(n):
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        dW = np.sqrt(-2*np.log(u)*(1.0)/float(n))*np.cos(2*np.pi*v)
        W = W+dW
        X1.append((x-h)*np.exp(sigma*W-(1/2*sigma**2-r)*(i+1)/n))
        X2.append((x+h)*np.exp(sigma*W-(1/2*sigma**2-r)*(i+1)/n))
        X3.append((x)*np.exp(sigma*W-(1/2*sigma**2-r)*(i+1)/n))
    return np.array([X1, X2, X3])


def indicatrice(value, range1, range2, open1=False, open2=False):
    if open1 == False and open2 == False:
        if value >= range1 and value <= range2:
            return 1
        else:
            return 0
    elif open1 == False and open2 == True:
        if value >= range1 and value < range2:
            return 1
        else:
            return 0
    elif open1 == True and open2 == False:
        if value > range1 and value <= range2:
            return 1
        else:
            return 0
    else:
        if value > range1 and value < range2:
            return value
        else:
            return 0


def H(s, k, delta):
    return indicatrice(s, k+delta, np.inf, open1=True, open2=False) + indicatrice(s, k-delta, k+delta, open1=False, open2=False)*(s-k+delta)/(2*delta)


def G(t, k, delta):
    return quad(H, -np.inf, t, args=(k, delta))[0]


def F(t, k, delta):
    return np.maximum(t-k, 0) - G(t, k, delta)


def I_(u, k, delta):
    if u >= k-delta and u <= k+delta:
        return (1/(2*delta))
    else:
        return 0


def F_(t, k, delta):
    return np.maximum(t-k, 0) - dblquad(I_, -np.inf, t, lambda x: -np.inf, lambda x: x, args=[delta])[0]
