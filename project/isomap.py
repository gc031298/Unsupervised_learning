import numpy as np
from scipy.spatial.distance import cdist

def nearest_neighbors(data, k):
    """
    Returns the k nearest neighbours of each data point.
    Output size: (n,k)
    """
    distance = cdist(data, data, metric = 'euclidean')
    return np.array([np.argsort(distance[i,:]) for i in range(np.shape(data)[0])]).astype(int)[:,1:k+1]


def nn_graph(data, neighbors):
    """
    Returns the graph connecting the k nearest neighbors
    """
    distance = cdist(data, data, metric = 'euclidean')
    graph = np.array([[0 if i == j else distance[i,j] if j in neighbors[i,:] else np.inf for j in range(np.shape(data)[0])] for i in range(np.shape(data)[0])])
    return graph


def floyd_warshall(graph):
    """
    Performs the Floyd-Warshall algorithm
    """
    n = np.shape(graph)[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                graph[i,j] = min(graph[j,i], graph[i,j], graph[i,k] + graph[k,j])
    if np.isinf(graph).any():
        raise Exception('The graph is not connected, not suitable for isomap. Increase the number of neighbors.')
    return graph


def isomap_double_centering(D):
    """
    Performs double centering of a matrix D
    """
    n = np.shape(D)[0]
    C = np.eye(n) - np.ones((n,n))/n
    D2 = D**2
    return -0.5*np.dot(C, np.dot(D2, C))


def isomap(data, d, k):
    """
    Performs isomap.
    Input:
    - data
    - d: dimension of the projection space
    - k: number of neighbours
    """
    m = np.shape(data)[1]
    if (d > m):
        raise Exception('Trying to project in a higher dimensional space. "d" must be smaller than the dimension of the data.')

    neighbors = nearest_neighbors(data,k)
    G = nn_graph(data, neighbors)
    F = floyd_warshall(G)
    F_centered = isomap_double_centering(F)

    lbd, eigenvectors = np.linalg.eigh(F_centered)
    idx = np.argsort(lbd)[::-1]

    return np.dot(eigenvectors[:,idx[0:d]],np.sqrt(np.diag(lbd[idx[0:d]])))