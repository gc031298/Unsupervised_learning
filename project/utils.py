import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors as NN
import warnings
warnings.filterwarnings("ignore")

# DATASETS

def swiss_roll(n):
    """
    Generate the swiss roll dataset.
    """
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        t = 8*i/n +2
        x[i]= t*np.sin(t)
        y[i]= t*np.cos(t)
        z[i]= np.random.uniform(-1, 1, 1)*6
    data = np.column_stack((x, y, z))
    return data


def s_curve(n):
    """
    Generate the s-curve dataset.
    """
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        t = (2.2*i-0.1)*np.pi/n
        z[i] = t
        y[i] = 3*np.sin(t)
        x[i] = np.random.uniform(-1, 1, 1)
    data = np.column_stack((x, y, z))
    return data

# UTILITIES

def nearest_neighbors(data, k):
    """
    Return the k-nearest neighbours of each data point.
    Output is a matrix of indexes with shape n x k.
    """
    distance = cdist(data, data, metric='euclidean')
    idx_neighbors = np.array([np.argsort(distance[i,:])[1:k+1]for i in range(np.shape(data)[0])]).astype(int)
    return idx_neighbors


def two_NN(data):
    """
    Return the intrinsic dimension of the manifold
    in which the data lie.
    Based on maximum likelihood estimation.
    """
    neigh = NN(n_neighbors=2).fit(data)
    distances, _ = neigh.kneighbors(return_distance=True)
    distances = distances[(distances > 1.e-5)[:,0],:]
    mu = distances[:,1]/distances[:,0]
    return data.shape[0]/np.sum(np.log(mu))


def compute_delta(data, k):
    """
    Compute distances between data points
    and their k-nearest neighbors.
    Output is a n x k matrix.
    """
    distance = cdist(data, data, metric='euclidean')
    delta = np.array([np.sort(distance[i,:])[1:k+1]for i in range(np.shape(data)[0])])
    return delta


def compute_theta(data, neighbors):
    """
    Compute angles between data points
    and their k-nearest neighbors, and
    collinear elements.
    Output is two n x k matrices, the 
    first for the angles and the latter
    for collinear elements.
    """
    k = neighbors.shape[1]
    theta = np.zeros((data.shape[0], k))
    collinear = np.zeros((data.shape[0], k), dtype=int)
    for i in range(data.shape[0]):
        for j in range(k):
            n = int(neighbors[i, j])
            v1 = data[i,:] - data[n,:]
            angles = np.zeros(data.shape[0])
            for z in range(k):
                m = int(neighbors[n, z])
                if m != i:
                    v2 = data[m,:] - data[n,:]
                    angles[m] = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            collinear[i,j] = np.argmin(np.abs(angles - np.pi))
            theta[i,j] = angles[collinear[i,j]]
    return theta, collinear


def pca(data, n_components):
    """
    Principal component analysis.
    Project data in a n_components dimensional space.
    """
    if (n_components > np.shape(data)[0]):
        raise Exception('Projecting in a space of higher dimension. Decrease n_components.')
    
    data = data - np.mean(data, axis = 0)
    cov = np.cov(data.T)
    eigenvalues, U = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    new_data = np.dot(data, U[:,idx[0:n_components]])
    return new_data, idx