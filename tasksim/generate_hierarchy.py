import numpy as np
from .tasksim import task_similarity

from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.cluster import AutoGMMCluster as GMM

from joblib import Parallel, delayed


def _generate_function_tuples(classes, metric_kwargs={'n_neg_classes':5}, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
       
    function_tuples = []
    for j, class1 in enumerate(classes):
        class1_idx = np.where(classes == class1)[0][0]
        for k, class2 in enumerate(classes):
            if j == k:
                continue

            class2_idx = np.where(classes == class2)[0][0]
    
            if metric_kwargs is not None:
                if 'n_neg_classes' in list(metric_kwargs.keys()):
                    for _ in range(n_neg_classes):
                        
                        neg_class_idx = np.random.choice(np.delete(classes, [j,k]) size=1)[0]
                        function_tuples.append((class1_idx, class2_idx, neg_class_idx))
            else:
                function_tuples.append(class1_idx, class2_idx)
    
    return function_tuples


def _array_to_matrix(a, n_classes, n_iterations_per_pair_of_classes):
    matrix = np.zeros((n_classes, n_classes))
    
    for j in range(n_classes):
        for k in range(n_classes):
            if j == k:
                matrix[j,k] = 0
                continue

            temp_index = j*n_classes + k - np.sum(np.arange(1, j+2))
            matrix[j,k] = np.mean(a[temp_index*n_iterations_per_pair_of_classes: (temp_index + 1)*n_iterations_per_pair_of_classes])
            
    return matrix


def task_sim_neg(class1, class2, negclass):
    n1, d = class1.shape
    n2, p = class2.shape
    n3, q = negclass.shape
    
    data1 = (np.concatenate([class1, negclass]), np.concatenate([np.zeros(n1), np.ones(n3)]))
    data2 = (np.concatenate([class2, negclass]), np.concatenate([np.zeros(n2), np.ones(n3)]))
    
    ts = task_similarity(data1, data2)
    
    return ts


def generate_dist_matrix(X, y, metric='tasksim', metric_kwargs={'n_neg_classes': 5}, function_tuples=None, n_cores=1, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
        
    classes = np.unique(y)
    idx_by_class = [np.where(y == c)[0] for c in classes]
        
    if metric == 'tasksim':
        if function_tuples is None:
            function_tuples = _generate_function_tuples(classes, metric_kwargs=metric_kwargs)
        
        condensed_func = lambda x: task_sim_neg(X[idx_by_class[x[0]]], X[idx_by_class[x[1]]], X[idx_by_class[x[2]]])
        n_iterations_per_pair_of_classes = len(classes)**2 - len(classes) 
    
    distances = np.array(Parallel(n_jobs=n_cores)(delayed(condensed_func)(tuple_) for tuple_ in function_tuples))
    dist_matrix = _array_to_matrix(distances, len(classes), n_iterations_per_pair_of_classes)
    
    return dist_matrix


def preprocess_dist_matrix(dist_matrix, make_symmetric=False, scale=False, aug_diag=False):
    if make_symmetric:
        dist_matrix = 0.5*(dist_matrix + dist_matrix.T)
        
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
