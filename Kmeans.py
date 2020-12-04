"""
K-Means Implementation
import using:
from Kmeans import Kmeans
"""
import torch
import numpy as np
import random
    

def Kmeans(x, K = 8):
    # Assumes x is NxD and y is size D
    # returns clusters_idx: a pytorch tensor of size N. It has indices of which cluster the point belongs to.
    # ie. clusters_idx[i] = 3 means x[i] belongs to cluster 3
    
    # initialization step for the loop
    x_s = x
    indices = torch.arange(x.shape[0], device = x.device)
    clusters_idx = torch.zeros_like(indices)
    check_count = 0
    max_iter = 100
    
    # initialize random K centers
    rand_idx = torch.randperm(x.shape[0], device = x.device)
    centers = x[rand_idx[:K]]
    
    while True:
        # reshape to allow distance calculations
        centers = centers.unsqueeze(dim = 1)
        x_s = x_s.unsqueeze(dim = 0)

        # get distance between all x and the centers:
        diff = x_s - centers
        dist = (diff * diff).sum(dim = 2)

        # return to original shapes:
        x_s = x_s.squeeze()
        centers = centers.squeeze()

        # sort new clusters based on distance to centers
        prev_clusters_idx = clusters_idx
        _, clusters_idx = torch.min(dist, dim = 0)
        
        
        # check if anything changed. If not we're done!
        if(torch.equal(clusters_idx, prev_clusters_idx)):
            break
        
        
        for k in range(K):
            mask = clusters_idx == k
            # calculate new centers at mean of clusters
            centers[k] = x_s[indices[mask]].mean()
        
        
        # if reached max iterations. Stop loop.
        check_count += 1
        if(check_count == max_iter):
            break
    
    return clusters_idx, centers

def Kmeans_testdata(x,centers):
    x_s = x
    
    # reshape to allow distance calculations
    centers = centers.unsqueeze(dim = 1)
    x_s = x_s.unsqueeze(dim = 0)

    # get distance between all x and the centers:
    diff = x_s - centers
    dist = (diff * diff).sum(dim = 2)

    # return to original shapes:
    x_s = x_s.squeeze()
    centers = centers.squeeze()

    # sort new clusters based on distance to centers
    _, clusters_idx = torch.min(dist, dim = 0)
    
    return clusters_idx