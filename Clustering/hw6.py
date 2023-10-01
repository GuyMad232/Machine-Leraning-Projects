import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for _ in range(k):
        r = np.random.randint(0, X.shape[0]-1)
        centroids.append(X[r])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 

    

def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    for k in range(len(centroids)):
        d = np.sum(np.absolute(X - centroids[k])**p, axis=1) ** (1/p)
        distances.append(d)
    distances = np.array(distances)
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    prev = np.array(centroids)
    for _ in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = distances.argmin(axis=0)
        
        for c in range(centroids.shape[0]):
            class_k = X[np.where(classes == c)]
            centroids[c] = np.mean(class_k , axis=0)
        
        if np.allclose(prev,centroids, rtol=0.07):
            break
        else:
            np.copyto(prev, centroids)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
        
def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # K Means++ initialization
    
    data = np.copy(X)
    centroids = []
    chosen_centroid = data[np.random.randint(data.shape[0]), :]
    centroids.append(chosen_centroid)
    for i in range(k-1):
        data = np.delete(data, chosen_centroid, axis=0)   
#         dist = np.sum(np.absolute(data - chosen_centroid)**p, axis=1) ** (1/p)
        # Calculate the distance from each point to its closest centroid
        distances = []    
        for c in range(len(centroids)):
            dist = np.sum(np.absolute(data - centroids[c])**p, axis=1) ** (1/p)
            distances.append(dist)
        distances = np.array(distances)
        dist = distances.min(axis=0)
        
        # Choose the next random centroid with weighted probability
        w = (dist ** 2) / np.sum((dist ** 2))
        next_centroid = np.random.choice(len(w), p=w)
        centroids.append(data[next_centroid,:])
        chosen_centroid = data[next_centroid,:]
    
    # K Means
    centroids = np.array(centroids).astype(np.float)
    prev = np.array(centroids)
    for _ in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = distances.argmin(axis=0)
        
        for j in range(centroids.shape[0]):
            class_k = X[np.where(classes == j)]
            centroids[j] = np.mean(class_k , axis=0)
        
        if np.allclose(prev,centroids, rtol=0.07):
            break
        else:
            np.copyto(prev, centroids)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
