import numpy as np
from .tasksim import task_similarity

from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.cluster import AutoGMMCluster as GMM

from joblib import Parallel, delayed

from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.cluster import AutoGMMCluster as GMM

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances

def _generate_function_tuples(classes, metric_kwargs={'n_neg_classes':5}, directed=True, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
       
    function_tuples = []
    
    if directed:
        for j, class1 in enumerate(classes):
            class1_idx = np.where(classes == class1)[0][0]
            for k, class2 in enumerate(classes):
                
                class2_idx = np.where(classes == class2)[0][0]

                if metric_kwargs is not None:
                    if 'n_neg_classes' in list(metric_kwargs.keys()):
                        n_neg_classes = metric_kwargs['n_neg_classes']
                        for _ in range(n_neg_classes):

                            neg_class_idx = np.random.choice(np.delete(classes, [j,k]), size=1)[0]
                            function_tuples.append((class1_idx, class2_idx, neg_class_idx))
                    if 'gamma' in list(metric_kwargs.keys()):
                        function_tuples.append((class1_idx, class2_idx, metric_kwargs['gamma']))
                else:
                    function_tuples.append((class1_idx, class2_idx))
    else:
        for j, class1 in enumerate(classes):
            class1_idx = np.where(classes == class1)[0][0]
            for k, class2 in enumerate(classes[j+1:]):

                class2_idx = np.where(classes == class2)[0][0]

                if metric_kwargs is not None:
                    if 'n_neg_classes' in list(metric_kwargs.keys()):
                        n_neg_classes = metric_kwargs['n_neg_classes']
                        for _ in range(n_neg_classes):

                            neg_class_idx = np.random.choice(np.delete(classes, [j,k]), size=1)[0]
                            function_tuples.append((class1_idx, class2_idx, neg_class_idx))
                    if 'gamma' in list(metric_kwargs.keys()):
                        function_tuples.append((class1_idx, class2_idx, metric_kwargs['gamma']))
                else:
                    function_tuples.append((class1_idx, class2_idx))
        
    
    return function_tuples


def _array_to_matrix(a, n_classes, n_iterations_per_pair_of_classes, directed):
    matrix = np.zeros((n_classes, n_classes))
    
    if directed:
        for j in range(n_classes):
            for k in range(n_classes):
                if j == k:
                    matrix[j,k] = 0
                    continue

                temp_index = j*n_classes + k
                temp_indices = np.arange(n_iterations_per_pair_of_classes * temp_index, 
                                        n_iterations_per_pair_of_classes * (temp_index+1))

                matrix[j,k] = np.mean(a[temp_indices])
    else:
        for j in range(n_classes):
            for k in range(n_classes):
                if j == k:
                    matrix[j,k] = 0
                    continue
                    
                if n_iterations_per_pair_of_classes == 1:
                    matrix[j,k] = a[j*n_classes + k]
                    matrix[k,j] = matrix[j,k]
                    
    return matrix


def task_sim_neg(class1, class2, negclass, task_similarity_kwargs={}):
    n1, d = class1.shape
    n2, p = class2.shape
    n3, q = negclass.shape
    
    data1 = (np.concatenate([class1, negclass]), np.concatenate([np.zeros(n1), np.ones(n3)]))
    data2 = (np.concatenate([class2, negclass]), np.concatenate([np.zeros(n2), np.ones(n3)]))
    
    ts = task_similarity(data1, data2, **task_similarity_kwargs)
    
    return ts

def mmd_rbf(X, Y, gamma='median'):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
        
    from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    """
    n, d = X.shape
    m, _ = Y.shape
    
    if gamma == 'median':
        gammax = np.median(np.sort(pairwise_distances(X).reshape(n**2,))[n:])
        gammay = np.median(np.sort(pairwise_distances(Y).reshape(n**2,))[n:])
        gammaxy = np.median(np.sort(pairwise_distances(X,Y).reshape(n*m,)))
    else:
        gammax, gammay, gammaxy = gamma, gamma, gamma
        
    XX = rbf_kernel(X, X, gammax) - np.eye(n)
    YY = rbf_kernel(Y, Y, gammay) - np.eye(m)
    XY = rbf_kernel(X, Y, gammaxy)
    
    return (XX.sum() / (n * (n-1))) + (YY.sum() / (m * (m-1))) - (2 * XY.mean())


def generate_dist_matrix(X, y, metric='tasksim', metric_kwargs={'n_neg_classes': 5}, function_tuples=None, n_cores=1, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
        
    classes = np.unique(y)
    idx_by_class = [np.where(y == c)[0].astype(int) for c in classes]
        
    if metric == 'tasksim':
        directed=True
        if function_tuples is None:
            function_tuples = _generate_function_tuples(classes, metric_kwargs, directed)
        
        condensed_func = lambda x: task_sim_neg(X[idx_by_class[int(x[0]])], X[idx_by_class[int(x[1]])], X[idx_by_class[int(x[2]])])
        
    elif metric == 'mmd':
        directed=True
        if function_tuples is None:
            function_tuples = _generate_function_tuples(classes, metric_kwargs, directed)
            
        condensed_func = lambda x: mmd_rbf(X[idx_by_class[x[0]]], X[idx_by_class[x[1]]], x[2])
        
    if directed:
        n_iterations_per_pair_of_classes = int(len(function_tuples) / (len(classes)**2))
    else:
        # This is broken
        n_iterations_per_pair_of_classes = int(len(function_tuples) / ((len(classes)**2 - len(classes)) / 2))
    
    distances = np.array(Parallel(n_jobs=n_cores)(delayed(condensed_func)(tuple_) for tuple_ in function_tuples))
    dist_matrix = _array_to_matrix(distances, len(classes), n_iterations_per_pair_of_classes, directed)
    
    return dist_matrix

def preprocess_dist_matrix(dist_matrix, make_symmetric=False, scale=False, aug_diag=False, negate=False):
    if make_symmetric:
        dist_matrix = 0.5*(dist_matrix + dist_matrix.T)
        
    if negate:
        dist_matrix = 1 - dist_matrix
        
    if aug_diag:
        n, _ = dist_matrix.shape
        
        for i in range(n):
            dist_matrix[i,i] = np.sum(dist_matrix[i]) / (n - 1)
        
    if scale:
        dist_matrix = (dist_matrix - np.min(dist_matrix)) / (np.max(dist_matrix) - np.min(dist_matrix))
                                    
    return dist_matrix


def cluster_dists(dist_matrix, embedding=ASE, embedding_kwargs={}, cluster=GMM, cluster_kwargs={}):
    if embedding is not None:
        X_hat = embedding(**embedding_kwargs).fit_transform(dist_matrix)
    else:
        X_hat = dist_matrix
         
    return cluster(**cluster_kwargs).fit_predict(X_hat)


def generate_hierarchy(X, y, 
                        generate_dist_matrix_kwargs,
                        process_dist_matrix_kwargs,
                        cluster_dists_kwargs,
                        acorn=None):
    
    if acorn is not None:
        np.random.seed(acorn)
    
    dist_matrix = generate_dist_matrix(X,y,**generate_dist_matrix_kwargs)
    processed_dist_matrix = preprocess_dist_matrix(dist_matrix, **process_dist_matrix_kwargs)
    clusters = cluster_dists(processed_dist_matrix, **cluster_dists_kwargs)
        
    return clusters
