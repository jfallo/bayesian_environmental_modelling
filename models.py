import numpy as np
from scipy.integrate import odeint
from scipy.stats import multinomial

def model_monod(par, C):
    """
    Model:
    ------
    Growth rate as a function of substrate concentration:
    r = r_max * C / (K + C)
    
    Arguments:
    ----------
    - par: array containing the following parameters:
        - par[0]: 'r_max' (maximum growth rate)
        - par[1]: 'K' (half-saturation concentration)
    - C: vector containing substrate concentrations
    
    Returns:
    -------
    r: A vector of growth rates
    """
    r_max = par[0]
    K = par[1]
    
    # calculate growth rate using the Monod model
    r = r_max * C / (K + C)
    
    return r


def model_growth(par, times):
    """
    Model:
    ------
    Growth of microorganisms on a substrate in a batch reactor:
    dC_M / dt = mu * C_S / (K + C_S) * C_M - b * C_M
    dC_S / dt = -1 / Y * mu * C_S / (K + C_S) * C_M
    
    Arguments:
    ----------
    - par: array containing the following parameters:
        [mu, K, b, Y, C_M_ini, C_S_ini]
    - times: array of time points to evaluate the ODE
    
    Value:
    ------
    A dictionary containing `C_M` and `C_S` for all points in `time`
    """
    
    # check if the correct number of parameters are provided
    if len(par) < 6:
        raise ValueError("Error in model_growth: wrong number of parameters provided.")

    def rhs_growth(C, t, par):
        """
        Define right-hand side of the ODE
        """
        
        # Extract state variables and parameters from the inputs
        C_M, C_S = C
        mu = par[0]  # mu: maximum growth rate of microorganisms
        K = par[1]  # K: half-concentration of growth rate with respect to substrate
        b = par[2]  # b: rate of death and respiration processes of microorganisms
        Y = par[3]  # Y: yield of growth process
        
        # calculate the rates of change
        r_M = mu * C_S / (K + C_S) * C_M - b * C_M  # = C_M * ((mu * C_S) / (K + C_S) - b)              microorganism concentration * (growth rate - death rate)
        r_S = -1 / Y * mu * C_S / (K + C_S) * C_M   # = C_M * ((mu * C_S) / (K + C_S)) * (-1 / Y)       microorganism concentration * growth rate * consumption
        
        return [r_M, r_S]
    
    C_M_ini = par[4]  # initial concentration of microorganisms
    C_S_ini = par[5]  # initial concentration of substrate
    C_ini = [C_M_ini, C_S_ini]
    
    # Solve the ODE system using odeint
    res_ode = odeint(rhs_growth, C_ini, times, args=(par,))
    
    return {'time': times, 'C_M': res_ode[:, 0], 'C_S': res_ode[:, 1]}


def model_survival(par, N, t):
    """
    Model:
    ------
    Population survival as a function of individual survival probability:
    S(t) = exp(-lambda * t)
    
    Arguments:
    ----------
    - par: array containing the following parameters:
        - par[0]: 'lambda' (mortality rate)
    - N: total number of individuals in the experiment
    - t: time points at which the number of deaths are counted
    
    Returns:
    -------
    y: A vector of within time interval death counts
    """
    lambda_ = par[0]

    # survival probablities at measurement points
    S = np.exp(-lambda_ * t)
    S = np.append(S, 0)

    # probabilities to die within time intervals
    p = -np.diff(S)

    # observe deatch counts
    y = multinomial.rvs(N, p)

    return y

