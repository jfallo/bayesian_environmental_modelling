import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# --- Sampling and evaluating random variables --- #

def f_X(x):
    return uniform.pdf(x)

def f_Y(y):
    return norm.pdf(y, loc= 2, scale= 10)

print(f"f_X(0.8) = {f_X(0.8)}")
print(f"f_Y(0.8) = {f_Y(0.8)}")


plt.figure(figsize= (14,6))

X = uniform.rvs(size= 1000)
plt.subplot(1,2,1)
sns.histplot(X, kde= False, bins= 30, color= 'blue', alpha= 0.6)
plt.title("Histogram of X Samples")
plt.xlabel("x")
plt.ylabel("frequency")

Y = norm.rvs(loc= 2, scale= 10, size= 1000)
plt.subplot(1,2,2)
sns.histplot(Y, kde= False, bins= 30, color= 'red', alpha= 0.6)
plt.title("Histogram of Y Samples")
plt.xlabel("y")
plt.ylabel("frequency")

plt.tight_layout()
plt.show()


def transform_X_to_Z(X):
    return np.sin(2 * np.pi * X) * np.sqrt(X)

Z = transform_X_to_Z(X)
plt.plot()
sns.histplot(Z, kde= False, bins= 75, color= 'green', alpha= 0.6)
plt.title("Histogram of Z Samples")
plt.xlabel("z")
plt.ylabel("frequency")
plt.tight_layout()
plt.show()


# --- Generating, analzing, and visualizing data --- #

Y1 = np.random.normal(loc= [3,8], scale= [2,5], size= (1000, 2))
Y1_quartiles = np.percentile(Y1, [25,75], axis= 0)
Y1_range = np.diff(Y1_quartiles, axis= 0)[0]
print(f"Y1 interquartile range: {Y1_range}")
Y1_cov = np.cov(Y1, rowvar= False)
Y1_corr = np.corrcoef(Y1, rowvar= False)
print("Y1 cov matrix:")
print(Y1_cov)
print("Y1 corr matrix:")
print(Y1_corr)

Y2 = np.random.multivariate_normal(mean= [3,8], cov= [[4,8], [8,25]], size= 1000)
Y2_quartiles = np.percentile(Y2, [25,75], axis= 0)
Y2_range = np.diff(Y2_quartiles, axis= 0)[0]
print(f"Y2 interquartile range: {Y2_range}")
Y2_cov = np.cov(Y2, rowvar= False)
Y2_corr = np.corrcoef(Y2, rowvar= False)
print("Y2 cov matrix:")
print(Y2_cov)
print("Y2 corr matrix:")
print(Y2_corr)

# histograms of marginals
fig, axes = plt.subplots(2, 2, figsize= (14,6))
fig.suptitle("Histograms of Marginal Samples")

plt.subplot(2, 2, 1)
sns.histplot(Y1[:,0], kde= False, bins= 30, color= 'blue', alpha= 0.6)
plt.xlabel("y1.1")
plt.ylabel("frequency")

plt.subplot(2, 2, 2)
sns.histplot(Y1[:,1], kde= False, bins= 30, color= 'blue', alpha= 0.6)
plt.xlabel("y1.2")
plt.ylabel("frequency")

plt.subplot(2, 2, 3)
sns.histplot(Y2[:,0], kde= False, bins= 30, color= 'blue', alpha= 0.6)
plt.xlabel("y2.1")
plt.ylabel("frequency")

plt.subplot(2, 2, 4)
sns.histplot(Y2[:,1], kde= False, bins= 30, color= 'blue', alpha= 0.6)
plt.xlabel("y2.2")
plt.ylabel("frequency")

plt.tight_layout()
plt.show()

# densities of marginals
fig, axes = plt.subplots(2, 2, figsize= (14,6))
fig.suptitle("Densities of Marginal Samples")

plt.subplot(2, 2, 1)
sns.kdeplot(Y1[:,0], color= 'blue', alpha= 0.6)
plt.xlabel("y1.1")
plt.ylabel("density")

plt.subplot(2, 2, 2)
sns.kdeplot(Y1[:,1], color= 'blue', alpha= 0.6)
plt.xlabel("y1.2")
plt.ylabel("density")

plt.subplot(2, 2, 3)
sns.kdeplot(Y2[:,0], color= 'blue', alpha= 0.6)
plt.xlabel("y2.1")
plt.ylabel("density")

plt.subplot(2, 2, 4)
sns.kdeplot(Y2[:,1], color= 'blue', alpha= 0.6)
plt.xlabel("y2.2")
plt.ylabel("density")

plt.tight_layout()
plt.show()

# scatterplots
fig, axes = plt.subplots(1, 2, figsize= (14,6))
fig.suptitle("Joint Scatterplots")

sns.scatterplot(x= Y1[:,0], y= Y1[:,1], alpha= 0.6, ax= axes[0])
axes[0].set_xlabel("y1.1")
axes[0].set_ylabel("y1.2")

sns.scatterplot(x= Y2[:,0], y= Y2[:,1], alpha= 0.6, ax= axes[1])
axes[1].set_xlabel("y2.1")
axes[1].set_ylabel("y2.2")

plt.tight_layout()
plt.show()


# --- Working with dataframes --- #

growth_data = pd.read_csv("data/model_growth.csv", sep= ' ')

t = growth_data['t']
C_M = growth_data['C_M']
C_S = growth_data['C_S']

plt.figure(figsize= (14,6))
sns.lineplot(x= t, y= C_M, marker= '^', color= 'red', label= 'Microorganisms')
sns.lineplot(x= t, y= C_S, marker= 'o', label= 'Substrate')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend(loc= 'upper right')
plt.tight_layout()
plt.show()

# add column
n = len(growth_data)
growth_data['C_new'] = np.random.normal(size= n)
print(growth_data.head())
