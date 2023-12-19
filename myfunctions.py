import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.linear_model import LinearRegression as LR
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt


# Laboratory 1


def swiss_roll(n):
    """
    Defines the swiss roll dataset
    """
    data = np.zeros((n,3))
    phi = np.random.uniform(low=1.5*np.pi, high=4.5*np.pi, size=n)
    psi = np.random.uniform(0,10,n)  
    data[:,0]=phi*np.cos(phi)
    data[:,1]=phi*np.sin(phi)
    data[:,2]=psi
    return data


def swiss_roll_gaussian_noise(n, mu = 0, sigma = 0.5):
    """
    Swiss roll data points with gaussian noise
    """
    data = np.zeros((n,3))
    phi = np.random.uniform(low=1.5*np.pi, high=4.5*np.pi, size=n)
    psi = np.random.uniform(0,10,n)
    noise = np.random.multivariate_normal(mean=[mu,mu], cov=sigma*np.eye(2), size=n)
    data[:,0]=phi*np.cos(phi) + noise[:,0]
    data[:,1]=phi*np.sin(phi) + noise[:,1]
    data[:,2]=psi
    return data


# Laboratory 2


def pca(data, n_components):
    """
    Performs PCA on a data set
    Input:
    - data
    - dimension of projection space
    Output: projected data as a np_array
    """
    if (n_components > np.shape(data)[0]):
        raise Exception('Projecting in a space of higher dimension. Decrease n_components.')
    
    data = data - np.mean(data, axis = 0)
    cov = np.cov(data.T)
    eigenvalues, U = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    new_data = np.dot(data, U[:,idx[0:n_components]])
    return new_data


def center_data(data):
    """
    Centers the data
    """
    return data - np.mean(data, axis = 0)


def normalize_data(data):
    """
    Normalizes the data
    """
    return (data - np.mean(data, axis = 0))/np.std(data, axis = 0)


# Laboratory 3


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


# Laboratory 4

def kernel_matrix(data, choice = 'rbf', sigma = 1, d = 0):
    """
    Computes the kernel matrix using either the rbf or the polynomial kernel.
    """
    data = np.array(data)
    n = np.shape(data)[0]
    K = np.zeros((n,n))
    if choice == 'rbf':
        K = np.exp(-sigma**(-2)*cdist(data, data, metric = 'euclidean')**2)
    else:
        K = (np.ones((n,n)) + np.dot(data, data.T))**d
    return K


def kernel_double_centering(K):
    """
    Performs double centering of matrix A.
    """
    n = np.shape(K)[0]
    I = np.ones((n,n))
    return K - np.dot(I,K)/n - np.dot(K,I)/n + np.dot(I, np.dot(K,I))/(n*n)


def kernel_PCA(data, n_components, choice = 'rbf', sigma = 1, d = 1):
    """
    Performs kernel PCA on a data set.
    Input:
    - data
    - dimension of projection space
    Output: projected data as a np_array.
    """
    if choice == 'rbf':
        K = kernel_matrix(data, 'rbf', sigma, d)
    else:
        K = kernel_matrix(data, 'polynomial', sigma, d)
    K_tilde = kernel_double_centering(K)
    eigs, U = np.linalg.eigh(K_tilde)
    idx = np.argsort(eigs)[::-1]
    return np.dot(U[:, idx[0:n_components]], np.diag(np.sqrt(eigs[idx[0:n_components]])))


# Laboratory 6


def two_NN(data, choice: str, discard_fraction = 0.1):
    """
    Given a dataset, returns the intrinsic dimension using 2-nn method.
    Input:
    - data: dataset
    - choice: how to compute the intrinsic dimensionality. {'ML', 'CDF'}
    """
    neigh = NN(n_neighbors=2).fit(data)
    distances, _ = neigh.kneighbors(return_distance=True)
    distances = distances[(distances > 1.e-5)[:,0],:]
    mu = distances[:,1]/distances[:,0]

    if choice == 'ML':
        return data.shape[0]/np.sum(np.log(mu))
    
    if choice == 'CDF':
        cdf = np.arange(mu.shape[0])/mu.shape[0]
        mu = mu[np.argsort(mu)]
        lr = LR(fit_intercept = False)
        lr.fit(np.log(mu).reshape(-1, 1), -np.log(1-cdf).reshape(-1, 1))
        return lr.coef_[0][0]

    raise Exception("Insert a valid choice parameter. Select among 'ML' and 'CDF'.")


def histogram(x):
    """
    Plots the histogram estimate of the probability density function of the data.
    """
    n = x.shape[0]
    x = sorted(x)
    q3, q1 = np.percentile(x, [75,25])
    iqr = q3 - q1
    h = 2*iqr*np.power(n, -1/3)
    m = int((x[-1] - x[0])/h)
    hist = np.zeros(m+1)

    for j in range(n):
        i = int((x[j] - x[0])/h)
        hist[i] += 1
    
    hist = hist/n
    left = np.array([(x[0] + j*h) for j in range(m+1)])
    plt.bar(left, hist, align = 'edge', color = 'steelblue')
    plt.grid()


def kde(x):
    """
    Computes the gaussian kernel density estimation of the probability density function of the data.
    """
    n = x.shape[0]
    sigma = np.std(x)
    q3, q1 = np.percentile(x, [75,25])
    iqr = q3 - q1
    h = 0.9*np.min([sigma, iqr/1.34])*np.power(n,-1/5)

    num = 100
    xx = np.linspace(-5, 15, num)

    K = np.array([[(np.exp(-0.5*((xx[j] - x[i])/h)**2)/np.math.sqrt(2*np.math.pi)) for j in range(num)] for i in range(n)])
    estimator = [(np.sum(K[:,l])/(n*h)) for l in range(n)]

    plt.plot(xx,estimator, color = 'forestgreen')
    plt.grid()


# Laboratory 7


def mutual_information(x,y):
    """
    Computes the mutual information between x and y.
    """
    xy = np.c_[x, y]
    values_x, count_x = np.unique(x, return_counts=True)
    values_y, count_y = np.unique(y, return_counts=True)
    values_xy, count_xy = np.unique(xy, return_counts=True, axis = 0)
    prob_x = [(count_x[i]/len(x)) for i in range(len(values_x))]
    prob_y = [(count_y[i]/len(y)) for i in range(len(values_y))]
    prob_xy = [(count_xy[i]/len(xy)) for i in range(len(values_xy))]

    sum = 0
    for j in range(len(values_xy)):
        idx_x = [index for index, value in enumerate(values_x) if (value == values_xy[j, 0])][0]
        idx_y = [index for index, value in enumerate(values_y) if (value == values_xy[j, 1])][0]
        sum += prob_xy[j]*np.log(prob_xy[j]/(prob_x[idx_x]*prob_y[idx_y]))
        
    return sum


def k_means(data, k, init = 'def'):
    """
    Performs clustering via k-means algorithm.
    Initialization:
    - def: k-means
    - k++: k-means++
    Returns:
    - cluster labels
    - total loss
    - centroids
    """
    if init == 'def':
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    elif init == 'k++':
        centroids = [data[np.random.choice(data.shape[0])]]
        for _ in range(1,k):
            min_dist = np.min(cdist(data, np.array(centroids), metric = 'euclidean'), axis = 1)
            new_centroid = data[np.random.choice(data.shape[0], p = min_dist**2 / np.sum(min_dist**2))]
            centroids.append(new_centroid)
        centroids = np.array(centroids)
    else:
        raise Exception("Select a valid initialization method: 'def' for k-means, 'k++' for k-means++.")
    
    while True:
        prev_centroids = centroids.copy()
        distances = cdist(data, centroids, metric='euclidean')
        labels = np.argmin(distances, axis=1)
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = np.mean(data[labels == i], axis=0)
        if np.allclose(prev_centroids, centroids):
            break
    
    loss = np.sum([np.linalg.norm(data[i] - centroids[labels[i]])**2 for i in range(len(data))])
    return labels, loss, centroids


# Laboratory 8


def k_medoids(data, k, init = 'def'):
    """
    Performs clustering via k-medoids algorithm.
    Initialization:
    - def: k-medoids
    - k++: k-medoids++
    Returns:
    - cluster labels
    - total loss
    - medoids
    """
    if init == 'def':
        medoids = data[np.random.choice(data.shape[0], k, replace=False)]
    elif init == 'k++':
        medoids = [data[np.random.choice(data.shape[0])]]
        for _ in range(1,k):
            min_dist = np.min(cdist(data, np.array(medoids), metric = 'euclidean'), axis = 1)
            new_medoid = data[np.random.choice(data.shape[0], p = min_dist**2 / np.sum(min_dist**2))]
            medoids.append(new_medoid)
        medoids = np.array(medoids)
    else:
        raise Exception("Select a valid initialization method: 'def' for k-medoids, 'k++' for k-medoids++.")
    
    while True:
        prev_medoids = medoids.copy()
        distances = cdist(data, medoids, metric='euclidean')
        labels = np.argmin(distances, axis=1)
        for i in range(k):
            if np.any(labels == i):
                medoids[i] = np.median(data[labels == i], axis=0)
        if np.allclose(prev_medoids, medoids):
            break
    
    loss = np.sum([np.linalg.norm(data[i] - medoids[labels[i]])**2 for i in range(len(data))])
    return labels, loss, medoids


def c_means(data, k, f=2, tolerance=1e-4):
    """
    Performs clustering via c-means algorithm.
    Returns:
    - cluster labels
    - total loss
    - centers
    """
    exp = 2/(f-1)
    U = np.random.rand(data.shape[0], k)
    U = U/np.sum(U, axis=1, keepdims=True)
    
    while True:
        prev_U = U.copy()
        weights = U**f
        centers = np.dot(weights.T, data)/np.dot(weights.T, np.ones((data.shape[0], 1)))
        distance = cdist(data, centers, metric='euclidean')
        distance = np.maximum(distance, 1.e-6) # avoid division by 0
        U = 1/np.sum((distance[:, :, None]/distance[:, None, :])**exp, axis = -1)
        if np.linalg.norm(U-prev_U) < tolerance:
            break

    labels = np.argmax(U, axis=1)
    loss = np.sum(weights*distance**2)
    return labels, loss, centers


# Laboratory 9


def spectral_clustering(data, n_neigh, k, init = 'def'):
    """
    Performs spectral clustering.
    Input:
    - data: matrix storing datapoints on rows
    - n_neigh: number of neighbors for similarity matrix
    - k: number of output clusters
    - init: 'def' for standard k-means,
            'k++' for k-means++. Default: 'def'
    Output:
    - labels: k cluster lables
    - loss: total k-means loss
    - centers
    """
    dist = cdist(data, data, metric='euclidean')
    neighbors = nearest_neighbors(dist, n_neigh)
    similarity = np.zeros_like(dist)
    for i in range(np.shape(data)[0]):
        for idx in neighbors[i,:]:
            if dist[i, int(idx)] != 0:
                similarity[i, int(idx)] = similarity[int(idx), i] = 1/dist[i, int(idx)]
    degree = np.diag(np.array([np.sum([similarity[i,j] for j in range(np.shape(data)[0])]) for i in range(np.shape(data)[0])]))
    laplacian = degree - similarity
    lbd, U = np.linalg.eigh(laplacian)
    idx = np.argsort(lbd)[:k]
    spectral_data = U[:,idx]
    labels, loss, _ = k_means(data = spectral_data, k = k, init = init)
    return labels, loss


# Laboratory 10


def density_peaks(data, k, dc):
    """
    Performs density peaks clustering.
    Args:
    - data: NxM matrix of data
    - k: number of clusters
    - dc: cutoff distance
    """
    n = np.shape(data)[0]
    distance = cdist(data, data, metric = 'euclidean')
    density = np.sum(np.exp(-(distance/dc)**2), axis = 1)
    idx_sorted = np.argsort(-density)
    
    delta = np.zeros(n)
    idx_neighbor = np.zeros(n, dtype=int)
    for i in range(1,n):
        delta[idx_sorted[i]] = np.min(distance[idx_sorted[i],idx_sorted[:i]])
        idx_neighbor[idx_sorted[i]] = idx_sorted[np.argmin(distance[idx_sorted[i],idx_sorted[:i]])]
    delta[idx_sorted[0]] = 1.05*np.max(delta)
    idx_centers = np.argsort(-delta)[:k]
    
    ind = [idx_sorted[i] for i in range(n)]
    labels = np.zeros(n, dtype = int)
    for j in range(k):
        labels[idx_centers[j]] = j+1
        ind.remove(idx_centers[j])
    
    cond = True
    while cond:
        for i in ind:
            if labels[idx_neighbor[i]] != 0:
                labels[i] = labels[idx_neighbor[i]]
                ind.remove(i)
        cond = len(ind) > 0
    
    return labels, data[idx_centers]


def normalized_mutual_information(x,y):
    """
    Computes normalized mutual information
    """
    def entropy(x):
        
        values, count = np.unique(x, return_counts=True)
        prob = [(count[i]/len(x)) for i in range(len(values))]
        entropy = -np.sum([prob[j]*np.log(prob[j]) for j in range(len(values))])
        return entropy
    
    return 2*mutual_information(x,y)/(entropy(x) + entropy(y))


def f_score(data, labels, centers):
    """
    Computes the F-score
    """
    n = np.shape(data)[0]
    k = np.shape(centers)[0]
    ssw = np.sum([np.linalg.norm(data[i] - centers[labels[i]-1])**2 for i in range(n)])
    _, count = np.unique(labels, return_counts=True)
    x_mean = np.mean(data, axis = 0)
    ssb = np.sum([count[j]*np.linalg.norm(centers[j] - x_mean)**2 for j in range(k)])
    return k*ssb/ssw