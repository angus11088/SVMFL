from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import torch
import math
from scipy.spatial import distance
from sklearn.decomposition import PCA

def standardize_data(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)

def apply_kmeans(x_scaled, n_clusters):
    model = KMeans(n_clusters=n_clusters, max_iter=50, tol=1e-3)
    return model.fit_predict(x_scaled)

def apply_gmm(x_scaled, n_components, covariance_type='full', tol=1e-3, max_iter=100, random_state=0):
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=tol, max_iter=max_iter, random_state=random_state)
    model.fit(x_scaled)
    return model.predict(x_scaled), model.predict_proba(x_scaled), model

def compute_mahalanobis(x, mean, cov_inv):
    return distance.mahalanobis(x, mean, cov_inv)
