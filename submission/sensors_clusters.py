import typing

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_pems_dataset():
    pems_data = np.load("PEMS08.npz")  
    print(f"\t- Data shape: {pems_data['data'].shape}") 
    return pems_data['data']

def parse_pems_dataset(pems_data: np.ndarray):
    flow, occupancy, speed = pems_data[:,:,0], pems_data[:,:,1], pems_data[:,:,2]
    sensor_df_all = [] # store all dfs of each sensor separately
    for i in range(pems_data.shape[1]):
        df = pd.DataFrame({
            "time_step": np.arange(pems_data.shape[0]),
            "sid": i,
            "speed": speed[:,i],
            "occupancy": occupancy[:,i],
            "flow": flow[:, i]
        })
        sensor_df_all.append(df)
    data = pd.concat(sensor_df_all, ignore_index=True, axis=0)
    data["dt_mins"] = pd.to_timedelta(data['time_step']*5, unit = "m")
    data['day'] = (data['dt_mins'].dt.days).astype(int)
    data["hour"] = (data["dt_mins"].dt.components.hours).astype(int)
    data_hourly = data.groupby(["sid", "day", "hour"], as_index= False)[["flow", "occupancy", "speed"]].mean()
    data_hourly['weekday'] = data_hourly['day'] % 7 # day of week, number coded
    data_weekday = data_hourly.groupby(["sid", "weekday", "hour"], as_index= False)[['flow', 'occupancy', 'speed']].mean()
    # pivot data to bring all features on a single sensor in 1 row
    data_weekly_tl = data_weekday.pivot(index="sid", columns = ["weekday", "hour"], values=['flow', 'occupancy', 'speed'])
    data_weekly_tl.columns = [f"{f}_wd{w}_hr{h}" for f,w,h in data_weekly_tl.columns]
    data_weekly_tl.reset_index(inplace=True)
    print(f"\t- Dataframe shape: {data_weekly_tl.shape}")
    return data_weekly_tl

def get_clean_features(data_weekly_tl: pd.DataFrame):
    X = data_weekly_tl.drop(columns="sid").to_numpy() # remove sensor id
    X = (X - X.mean(axis=0)) / X.std(axis=0) # standardization
    print(f"\t- Feature array shape: {X.shape}")
    return X

def get_pca_features(X:np.ndarray):
    pca = PCA().fit(X)
    cumvar = pca.explained_variance_ratio_.cumsum()
    print(f"\t- PCA result: {np.argmax(cumvar >= 0.99) + 1} components give 99% variance")
    pca = PCA(n_components= 45) #taking 45 components
    X_pca = pca.fit_transform(X)
    return X_pca

def calculate_clusters(X_pca: np.ndarray, n_c: int):
    mix, mu, sigma, resp, ll = gmm(X_pca, max_iter=100, n_clusters=n_c, init_method="random", seed= 42)
    labels = np.argmax(resp, axis=1)
    print(f"\t- Final log likelihood: {float(ll[-1])}")
    return mix, mu, sigma, labels
    
def display_results(mix:np.ndarray, mu:np.ndarray, sigma:np.ndarray, labels:np.ndarray, X_pca: np.ndarray):
    print("Results:\n--------------------")
    print(f"Cluster assignment as # of points:\n{pd.Series(labels).value_counts(ascending = True)}")

    print(f"\nMixing coefficients:\n{'  |   '.join([f'Cluster {i} -> {round(mix[i],3)}' for i in range(len(mix))])}")
    fig1, ax1 = plt.subplots(figsize=(8,8))
    sns.scatterplot(data = pd.DataFrame(X_pca[:, :2]), x= 0, y =1, hue = labels, palette="bright", s= 50, alpha=0.75, ax=ax1)
    ax1.scatter(mu[:,0], mu[:,1], color=["cyan", "magenta","yellow"], s=120, edgecolor='k', marker="o", label="Cluster Means")
    ax1.set_xlabel("PCA 1")
    ax1.set_ylabel("PCA 2")
    ax1.set_title("GMM Clusters on First Two PCA Components")
    ax1.grid(True)

    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 6))

    for i in range(sigma.shape[0]):
        sns.heatmap(sigma[i,:,:], ax = ax2[i], cmap="Spectral", square=True, linewidth=0.5)
        ax2[i].set_title(f"Covariance matrix for cluster {i+1}", fontsize= 12)
        ax2[i].tick_params(axis='both',
         which='major', labelsize=6)
        
    data_weekly_tl['cluster'] = labels
    feature_groups = ['flow', 'speed', 'occupancy']

    fig3, ax3 = plt.subplots(3, 1, figsize=(15, 8))

    for i in range(len(feature_groups)):
        feature = feature_groups[i]
        ax = ax3[i]
        
        cols = [c for c in data_weekly_tl.columns if c.startswith(feature)]
        cluster_means = data_weekly_tl.groupby('cluster')[cols].mean()

        cluster_means.columns = np.arange(len(cols))
        
        for cluster_id, row in cluster_means.iterrows():
            ax.plot(row.index, row.values, label=f'Cluster {cluster_id}')
        
        ax.set_title(f'Cluster-wise Mean {feature.capitalize()} Profile')
        ax.set_xlabel('Hours since start')
        ax.set_ylabel(feature)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def gmm(X: np.ndarray, max_iter: int, n_clusters: int, init_method: typing.Literal["kmeans", "random"] = "random", seed: int = 42):
    n, d = X.shape
    k = n_clusters
    prev_ll = -np.inf # initialize likelihood
    tol = 1e-6
    lls = []
    c_pi = np.ones(k) / k # mixing coeffs
    c_sigma = np.array([np.eye(d) for _ in range(k)]) # cluster cov initialized as identity matrices
    if init_method == 'kmeans':
        km = KMeans(n_clusters=k, random_state=seed, n_init=2).fit(X)
        c_mu = km.cluster_centers_.astype(float) # cluster means initialize from K Means cluster centers
    else:
        np.random.seed(seed)
        init_idx = np.random.choice(len(X), k, replace=False)
        c_mu = X[init_idx] # cluster means initialize from random datapoints
    for iter in range(max_iter):
        resp = np.zeros((n, k))
        log_resp = np.zeros((n,k))
        for j in range(k):
            gaussian = multivariate_normal(mean=c_mu[j], cov=c_sigma[j], allow_singular=True) # Normal Function
            log_resp[:, j] = np.log(c_pi[j]) + gaussian.logpdf(X) 
        log_resp_norm = logsumexp(log_resp, axis=1, keepdims=True) # take log of summation of exp of the logprobs for every point
        resp = np.exp(log_resp - log_resp_norm)  
        N_k = resp.sum(axis=0)  # total cluster responsibility for all points
        c_mu = np.einsum('nk,nd->kd', resp, X) / N_k[:, None] # update cluster mean
        c_sigma = np.zeros((k, d, d))
        for j in range(k):
            xdiff = X - c_mu[j]
            c_sigma[j] = np.einsum('n,ni,nj->ij', resp[:, j], xdiff, xdiff) / N_k[j] # update cluster cov
            c_sigma[j] += np.eye(d) * 1e-6
        c_pi = N_k / n # update mixing coefficients
        curr_log_likelihood = log_resp_norm.sum()
        lls.append(curr_log_likelihood)
        if np.abs(curr_log_likelihood - prev_ll) < tol: # check if not sufficient improvement in curr iter
            print(f"Termination criteria reached")
            break
        prev_ll = curr_log_likelihood
    return c_pi, c_mu, c_sigma, resp, lls



if __name__ == "__main__":
    print(">> Reading data | steps done: 1/5")
    pems_data = get_pems_dataset()
    print(">> Parsing data | steps done: 2/5")
    data_weekly_tl = parse_pems_dataset(pems_data)
    print(">> Preprocessing data | steps done: 3/5")
    X = get_clean_features(data_weekly_tl)
    print(">> Reducing data dimentions | steps done: 4/5")
    X_pca = get_pca_features(X)
    print(">> Calculating clusters | steps done: 5/5")
    mix, mu, sigma, labels = calculate_clusters(X_pca,3)
    display_results(mix, mu, sigma, labels, X_pca)
