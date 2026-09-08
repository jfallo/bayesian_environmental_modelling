from models import model_monod, model_growth, model_survival
import pandas as pd
import numpy as np
from scipy.stats import norm, multinomial
from scipy.optimize import minimize
import statsmodels.api as sm
from SALib.sample import fast_sampler
from SALib.analyze import fast
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

class MonodParam(Enum):
    r_max = 0
    K = 1


# --- Likelihood for a linear model --- #

def loglikelihood_linear(theta, x, y):
    """
    Calculate the log-likelihood for a linear model.

    Parameters:
    theta (list or array-like): Parameters of the model [slope, intercept, standard deviation].
    x (array-like): Independent variable.
    y (array-like): Dependent variable.

    Returns:
    float: The log-likelihood value.
    """
    beta, gamma, sigma = theta

    # Deterministic part
    y_det = beta * x + gamma

    # Calculate log-likelihood
    log_likelihood = np.sum(norm.logpdf(y, loc= y_det, scale= sigma))

    return log_likelihood

def neg_loglikelihood_linear(theta, x, y):
    return -loglikelihood_linear(theta, x, y)


data = pd.read_csv("data/model_linear.csv", sep= " ")

x = data['x']
y = data['y']
theta = [2, 1, 1]

print(f"Log-likelihood: {loglikelihood_linear(theta, x, y)}")


MLE = minimize(neg_loglikelihood_linear, theta, args= (x, y))
theta_MLE = MLE.x

beta, gamma, sigma = theta_MLE
y_pred = beta * x + gamma

print(f"MLE parameters: {theta_MLE}")
print()

plt.figure(figsize= (14,6))
plt.scatter(x, y, label= 'obs')
plt.plot(x, y_pred, color= 'red', label= 'pred model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()


x_with_constant = sm.add_constant(x)
model = sm.OLS(y, x_with_constant).fit()

print(model.summary())
print()


# --- Likelihood for Monod model --- #

def loglikelihood_Monod(theta, C, r):
    """
    Calculate the log-likelihood for the Monod model.

    Parameters:
    theta (list or array-like): Parameters of the model [r_max, K, sigma].
    x (array-like): Independent variable.
    y (array-like): Dependent variable.
    
    Returns:
    float: The log-likelihood value.
    """
    r_max, K, sigma = theta

    # Deterministic part
    r_det = (r_max * C) / (K + C)
    
    # Calculate log-likelihood
    log_likelihood = np.sum(norm.logpdf(r, loc= r_det, scale= sigma))

    return log_likelihood


data = pd.read_csv("data/model_monod_stoch.csv", sep= " ")

C = data['C']
r = data['r']

print(f"Log-likelihood for theta= [ 5, 3, 0.2]: {loglikelihood_Monod([5, 3, 0.2], C, r)}")
print(f"Log-likelihood for theta= [10, 4, 0.2]: {loglikelihood_Monod([10, 4, 0.2], C, r)}")

theta_monod = [5, 3, 0.2]
r_max, K, sigma = theta_monod


r_det = (r_max * C) / (K + C)

plt.figure(figsize= (14,6))
sns.scatterplot(x= C, y= r, label= 'obs')
sns.lineplot(x= C, y= r_det, color= 'red', label= 'det model')
plt.xlabel('substrate concentration (C)')
plt.ylabel('growth rate (r)')
plt.legend()
plt.tight_layout()
plt.show()


# --- Forward model simulation --- #

def simulate_monod_stoch(par, C):
    """
    Simulate the Monod model with stochastic noise.
    
    Arguments:
    ----------
    - par: Array containing the following parameters:
        - r_max: maximum growth rate
        - K: half-saturation concentration
        - sigma: standard deviation of noise
    - C: numpy array containing substrate concentrations
    
    Value:
    ------
    A numpy array representing the growth rate with stochastic noise added.
    """
    n = len(C)
    r_max, K, sigma = par

    r_det = model_monod([r_max, K], C)
    noise = np.random.normal(0, sigma, size= n)

    return r_det + noise


par_monod = [5, 3]
r_max, K = par_monod
dC = 0.1

C = np.arange(0, 10 + dC, dC)
r_det = model_monod(par_monod, C)

r_sim = np.array([simulate_monod_stoch(theta_monod, C) for _ in range(1000)])
quant = np.quantile(r_sim, q= [0.1, 0.9], axis= 0)

plt.figure(figsize= (14,6))
sns.lineplot(x= C, y= r_det, color= 'red', label= 'deterministic model', linewidth= 2)
plt.fill_between(C, quant[0,:], quant[1,:], color= 'gray', alpha= 0.5, label= '10th-90th percentile')
plt.title('Monod model outputs')
plt.xlabel('substrate concentration (C)')
plt.ylabel('growth rate (r)')
plt.legend(loc= 'upper right')
plt.tight_layout()
plt.show()


par_growth = [4, 10, 1, 0.6, 10, 50]
mu, K, b, Y, C_M_init, C_S_init = par_growth
dt = 0.01

times = np.arange(0, 2 + dt, dt)
out = model_growth(par_growth, times)

plt.figure(figsize= (14,6))
sns.lineplot(x= out['time'], y= out['C_M'], color= 'green', label= 'microorganisms')
sns.lineplot(x= out['time'], y= out['C_S'], color= 'blue', label= 'substrate')
plt.ylim(0, 51)
plt.title('Growth of microorganisms and substrate over time')
plt.xlabel('time')
plt.ylabel('concentration')
plt.legend(loc= 'upper right')
plt.tight_layout()
plt.show()


# --- Likelihood and forward simulation for the Survival model --- #

def loglikelihood_survival(y, par, N, t):
    """
    Calculate the log-likelihood of the observed data y given the parameters par
    and the times t using the multinomial distribution

    Arguments:
    ----------
    - y: observed data (number of deaths at each time point)
    - par: array containing the following parameters:
        - par[0]: 'lambda' (mortality rate)
    - N: total number of individuals in the experiment
    - t: time points at which the number of deaths are counted

    Returns:
    - LL: log-likelihood value
    """
    lambda_ = par[0]

    # survival probabilities at measurment points
    S = np.exp(-lambda_ * t)
    S = np.append(S, 0)

    # probabilities to die within time intervals
    p = -np.diff(S)

    return multinomial.logpmf(y, N, p)


N = 30
T = 5

t = np.arange(T+1)
par_survival = [0.2]
n_samples = 1000

deaths_pred = [
    model_survival(par_survival, N, t)[:-1]
    for _ in range(n_samples)
] 
deaths_pred_df = pd.DataFrame(deaths_pred, columns= t[1:])

plt.figure(figsize= (14,6))
sns.boxplot(data= deaths_pred_df)
plt.xlabel('time')
plt.ylabel('deaths')
plt.title('Death Counts within Time Intervals')
plt.tight_layout()
plt.show()

print(f"Log-likelihood of simulation: {loglikelihood_survival(model_survival(par_survival, N, t), par_survival, N, t)}")


# --- Sensitivity analysis --- #

# local analysis
def local_sa(model, par_vector, par, delta_rel= 0.1, *args, **kwargs):
    """
    Perform local sensitivity analysis on the model function.

    Parameters:
    - model: Function representing the model.
    - par_vector: Array of model parameters.
    - par: Name of the parameter to perform sensitivity analysis on.
    - delta_rel: Relative change in the parameter value.
    - *args, **kwargs: Additional arguments for the model function.

    Returns:
    Dictionary containing absolute and relative sensitivity indices.
    """
    # Copy the parameter array to adjust the parameter of interest
    par_vector2 = par_vector.copy()
    # Adjust the specified parameter by the relative delta
    par_vector2[par] *= (1 + delta_rel)

    # Run the model with the original and adjusted parameter array
    Y = model(par_vector, *args, **kwargs)
    Y2 = model(par_vector2, *args, **kwargs)

    # Compute sensitivity indices
    S_abs = (Y2 - Y) / (par_vector2[par] - par_vector[par])
    S_rel = (par_vector[par] / Y) * S_abs

    # Return sensitivity indices as a dictionary
    return {'abs': S_abs, 'rel': S_rel}


S = {'r_max': {}, 'K': {}}
delta_rels = [0.1, 0.5]
for delta_rel in delta_rels:
    S['r_max'][f'{delta_rel}'] = local_sa(model_monod, par_monod, MonodParam.r_max.value, delta_rel, C)
    S['K'][f'{delta_rel}'] = local_sa(model_monod, par_monod, MonodParam.K.value, delta_rel, C)

# absolute sensitivities
sns.set_theme(style= 'whitegrid')
plt.figure(figsize= (10,6))
sns.lineplot(x= C, y= S['K']['0.1']['abs'], color= 'red', linestyle= '-', label= 'S_K 10%')
sns.lineplot(x= C, y= S['K']['0.5']['abs'], color= 'red', linestyle= '--', label= 'S_K 50%')
sns.lineplot(x= C, y= S['r_max']['0.1']['abs'], color= 'blue', linestyle= '-', label= 'S_rmax 10%')
sns.lineplot(x= C, y= S['r_max']['0.5']['abs'], color= 'blue', linestyle= '--', label= 'S_rmax 50%')
plt.ylabel('absolute sensitivity')
plt.title('Absolute Sensitivities')
plt.legend(loc= 'upper left')
plt.tight_layout()
plt.show()

# relative sensitivities
sns.set_theme(style= 'whitegrid')
plt.figure(figsize= (10,6))
sns.lineplot(x= C, y= S['K']['0.1']['rel'], color= 'red', linestyle= '-', label= 'S_K 10%')
sns.lineplot(x= C, y= S['K']['0.5']['rel'], color= 'red', linestyle= '--', label= 'S_K 50%')
sns.lineplot(x= C, y= S['r_max']['0.1']['rel'], color= 'blue', linestyle= '-', label= 'S_rmax 10%')
sns.lineplot(x= C, y= S['r_max']['0.5']['rel'], color= 'blue', linestyle= '--', label= 'S_rmax 50%')
plt.ylim([-1.1, 1.1])
plt.ylabel('relative sensitivity')
plt.title('Relative Sensitivities')
plt.legend(loc= 'lower right')
plt.tight_layout()
plt.show()


# variance-based analysis
lower = np.array(list(par_monod)) * 0.7
upper = np.array(list(par_monod)) * 1.3
problem_dict = {
    'num_vars': 2,
    'names': ['r_max', 'K'],
    'bounds': list(zip(lower, upper))
}

par_monod_samples = fast_sampler.sample(problem_dict, N= 100)
Y_monod_from_samples = np.array([
    model_monod(par_monod_sample, C)
    for par_monod_sample in par_monod_samples
])

S = fast.analyze(problem_dict, Y= Y_monod_from_samples[:,4], print_to_console= True)

data = {
    'Factor': S['names'],
    'First-Order': S['S1'],
    'Total-Order': S['ST']
}
df = pd.DataFrame(data).melt(
    id_vars= 'Factor', 
    var_name= 'Sensitivity Type', 
    value_name= 'Index'
)

sns.barplot(data= df, x= 'Factor', y= 'Index', hue= 'Sensitivity Type')
plt.ylabel('Sensitivity Index')
plt.title('Sensitivity Analysis Results')
plt.show()
