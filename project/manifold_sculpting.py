import numpy as np
from matplotlib import pyplot as plt
import utils

class ManifoldSculpting():
    
    def __init__(self, data, n_neigh: int = 10, n_iter: int = 100, d: int = None, sigma: float = 0.95, save: bool = False, name: str = None):
        """
        Manifold Sculpting projects data to a lower dimensional space, while 
        preserving relations between datapoints simulating surface tension.
        
        Initialization:
        - data: data to be fitted
        - n_neigh: number of nearest neighbors
        - n_iter: number of iterations
        - d: dimension of the projection space
        - sigma: scale factor
        - save: flag for saving images
        - name: name for saving images
        """
        self.data = data
        self.n_neigh = n_neigh
        self.n_iter = n_iter
        self.sigma = sigma
        self.set_dimension(d)
        self.len = np.shape(self.data)[0]
        self.save = save
        self.name = name
            
                
    def set_dimension(self, d):
        """
        Set dimension of the projection space.
        If d is None, d is determined by TwoNN.
        """
        if d is None:
            self.d = round(utils.two_NN(self.data))
        else:
            self.d = d
            
            
    def fit(self):
        """
        Transform data.
        """
        # nearest neighbors
        self.neighbors = utils.nearest_neighbors(self.data, self.n_neigh)
        
        # distances, angles and average distance
        self.delta = utils.compute_delta(self.data, self.n_neigh)
        self.theta, self.collinear = utils.compute_theta(self.data, self.neighbors)
        self.av_delta = np.mean(self.delta)
        
        # pca rotation
        pca_data, idx = utils.pca(self.data, np.shape(self.data)[1])
        self.dpres = idx[:self.d]
        self.dscal = idx[self.d:]
        
        for k in range(self.n_iter):
            # point 4a -- rescaling
            pca_data[:,self.dscal] *= self.sigma
                
            # point 4b -- restoration
            rand_idx = np.random.randint(0, self.len, 1)
            queue = [rand_idx]
            adjusted = []
            while len(queue) > 0:
                idx = int(queue.pop(0))
                pca_data = self.adjust(pca_data, idx, adjusted)
                adjusted.append(idx)
                for el in self.neighbors[idx, :]:
                    if el not in adjusted and el not in queue:
                        queue.append(el)
            
            if self.save:
                print(f"Iteration: {k}, saving image...")
                if self.data.shape[1] <= 3:
                    self.get_representation(pca_data, k, self.name)
                    
        # point 5 -- projection
        self.projected_data = pca_data[:,self.dpres]
        self.transformed_data = pca_data  
    
    
    def adjust(self, data, idx, adjusted):
        """
        Adjust the data in order to minimize the relationships
        error. Minimization is performed by simple hill-climbing.
        """
        alpha = 0.3*self.av_delta
        condition = True
        while condition:
            error = self.compute_error(data, idx, adjusted)
            for k in self.dpres:
                data[idx,k] += alpha
                new_error = self.compute_error(data, idx, adjusted)
                condition = new_error < error
                if not condition:
                    data[idx,k] -= 2*alpha
                    new_error = self.compute_error(data, idx, adjusted)
                    condition = new_error < error
                    if not condition:
                        data[idx,k] += alpha
        return data
    
    
    def compute_error(self, data, idx, adjusted):
        """
        Compute an error to evaluate the current relations 
        among datapoints with respect to the original ones.
        """
        error = 0
        for i,j in enumerate(self.neighbors[idx]):
            weight = 1 + 9*(j in adjusted)
            new_delta = np.linalg.norm(data[idx]-data[j], 2)
            v1 = data[idx,:] - data[j,:]
            v2 = data[self.collinear[idx,i]] - data[j]
            new_angle = np.arccos(np.dot(v1.T,v2)/(np.linalg.norm(v1 ,2)*np.linalg.norm(v2 ,2)))
            err_delta = ((new_delta - self.delta[idx,i])/(2*self.av_delta))**2
            err_theta = ((new_angle - self.theta[idx,i])/np.pi)**2
            error += weight*(err_delta + err_theta)
        return error
            
            
    def get_representation(self, data, i, name):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,2], data[:,1], data[:,0], c=self.data[:,0])
        ax.set_xticklabels([])
        ax.set_yticklabels([]) 
        ax.set_zlim(-10,10)
        ax.view_init(elev=2,azim=25)
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        ax.set_zlabel('x')
        filename = 'images/' + name + '_' + str(i)+ '_iterations.png'
        plt.savefig(filename)
        plt.close(fig)