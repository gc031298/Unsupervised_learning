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
    - centered data
    - dimension of projection space
    Output: projected data as a np_array
    """
    if (n_components > np.shape(data)[0]):
        raise Exception('Projecting in a space of higher dimension. Decrease n_components.')
    
    cov = np.dot(data.T, data)
    eigenvalues = np.linalg.eig(cov)[0]
    U = np.linalg.eig(cov)[1]

    # # To have the same ouputs as sklearn.PCA:
    # for i in range(np.shape(U)[0]):
    #     j = np.argmax(np.abs(U[i,:]))
    #     if U[i,j] < 0:
    #         U[:,i] = -U[:,i]

    idx = np.argsort(eigenvalues)[::-1]
    new_data = np.dot(data, U[:,idx[0:n_components]])
    return new_data


def center_data(data):
    """
    Centers the data
    """
    data = np.array(data)
    n = np.shape(data)[1]
    for i in range(n):
        data[:,i] = data[:,i] - np.ones(np.shape(data)[0])*np.mean(data[:,i])
    return data


def normalize_data(data):
    """
    Normalizes the data
    """
    data = np.array(data)
    n = np.shape(data)[1]
    for i in range(n):
        data[:,i] = (data[:,i] - np.ones(np.shape(data)[0])*np.mean(data[:,i]))/np.std(data[:,i])
    return data


# Laboratory 3


def distance_matrix(x):
    """
    Returns the matrix of distances between datapoints
    """
    n = np.shape(x)[0]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            D[i,j] = D[j,i] = np.linalg.norm(x[i,:] - x[j,:])
    return D


def nearest_neighbors(D, k):
    """
    Returns the k nearest neighbours of each data point.
    Output size: (n,k)
    """
    n = np.shape(D)[0]
    neighbors = np.zeros((n,k))
    for i in range(n):
        neighbors[i,:] = np.argsort(D[i,:])[1:k+1]
    return neighbors


def nn_graph(D, neighbors):
    """
    Returns the graph connecting the k nearest neighbors
    """
    n = np.shape(D)[0]
    graph = np.ones_like(D) * np.inf
    for i in range(n):
        graph[i,i] = 0
        for idx in neighbors[i,:]:
            graph[i,int(idx)] = D[i,int(idx)]
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


def pow2_matrix(D):
    """
    Returns a new matrix with all components squared
    """
    n, m = np.shape(D)
    squared_D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            squared_D[i,j] = D[i,j] ** 2
    return squared_D


def isomap_double_centering(D):
    """
    Performs double centering of a matrix D
    """
    n = np.shape(D)[0]
    C = np.eye(n) - np.ones((n,n))/n
    D2 = pow2_matrix(D)
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
    
    D = distance_matrix(data)
    neighbors = nearest_neighbors(D,k)
    G = nn_graph(D, neighbors)
    F = floyd_warshall(G)
    F_centered = isomap_double_centering(F)

    lbd, eigenvectors = np.linalg.eigh(F_centered)
    idx = np.argsort(lbd)[::-1]

    return np.dot(eigenvectors[:,idx[0:d]],np.sqrt(np.diag(lbd[idx[0:d]])))


# Laboratory 4


def rbf_kernel(x1, x2, sigma):
    """
    Returns the computation of the Gaussian kernel between x1 and x2.
    """
    gamma = sigma**(-2)
    return np.exp(-gamma*np.linalg.norm(x1-x2)**2)


def polynomial_kernel(x1, x2, d):
    """
    Returns the compuation of the polynomial kernel of degree d between x1 and x2.
    """
    return (1 + np.dot(x1.T, x2))**d


def kernel_matrix(data, choice = 'rbf', sigma = 1, d = 0):
    """
    Computes the kernel matrix using either the rbf or the polynomial kernel.
    """
    data = np.array(data)
    n = np.shape(data)[0]
    K = np.zeros((n,n))
    if choice == 'rbf':
        K = [[(rbf_kernel(data[i,:], data[j,:], sigma = sigma)) for j in range(n)] for i in range(n)]
    else:
        K = [[(polynomial_kernel(data[i,:], data[j,:], d = d)) for j in range(n)] for i in range(n)]
    return K


def kernel_double_centering(K):
    """
    Performs double centering of matrix A.
    """
    n = np.shape(K)[0]
    I = np.ones((n,n))/n
    return K - np.dot(I,K) - np.dot(K,I) + np.dot(I, np.dot(K,I))


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
    eigs = np.linalg.eigh(K_tilde)[0]
    U = np.linalg.eigh(K_tilde)[1]

    for i in range(np.shape(U)[0]):
        j = np.argmax(np.abs(U[i,:]))
        if U[i,j] < 0:
            U[:,i] = -U[:,i]

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
    
    loss = np.sum([np.sum([(labels[i] == j)*(np.linalg.norm(data[i,:] - centroids[j,:])**2) for i in range(data.shape[0])]) for j in range(k)])
    return labels, loss


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
    
    loss = np.sum([np.sum([(labels[i] == j)*(np.linalg.norm(data[i,:] - medoids[j,:])**2) for i in range(data.shape[0])]) for j in range(k)])
    return labels, loss


def c_means(data, k, f=2, tolerance=1e-4):
    """
    Performs clustering via c-means algorithm.
    Returns:
    - cluster labels
    - total loss
    """
    exp = 2/(f-1)
    U = np.random.rand(data.shape[0], k)
    U /= np.sum(U, axis=1, keepdims=True)

    while True:
        prev_U = U.copy()
        centers = np.dot(U.T**f, data)/np.dot(U.T**f, np.ones((data.shape[0], 1)))
        #centers = np.dot(U.T**f, data)/np.sum(U ** f, axis=0, keepdims=True).T
        distance = cdist(data, centers, metric='euclidean')
        distance = np.maximum(distance, 1.e-6)  # Avoid division by zero
        distances_exp = (distance[:, :, None] / distance[:, None, :]) ** exp
        U = 1 / np.sum(distances_exp, axis=-1)
        if np.linalg.norm(U-prev_U) < tolerance:
            break

    labels = np.argmax(U, axis=1)
    loss = np.sum(U ** f * distance ** 2)
    return labels, loss