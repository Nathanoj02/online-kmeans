import cv2 as cv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "python"))

import pysignals

def elbow_method (img_path : str, start_k : int, end_k : int) :
    img = cv.imread(img_path)
    error_arr = np.array([])

    for k in range (start_k, end_k + 1) :
        res = pysignals.seq.k_means(img, k, 0.01)

        # Calculate sum of squared error
        error = np.sum((img - res) ** 2)
        error_arr = np.append(error_arr, error)

    # Plot
    plt.plot(np.arange(start_k, end_k + 1), error_arr, '-o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Total within sum of squares')
    plt.savefig('../data/stats/elbow.jpg')


def _dissimilarity(pixel1 : np.ndarray, pixel2 : np.ndarray) :
    return np.linalg.norm(np.array(pixel1) - np.array(pixel2))


def silouette_method (img_path : str, start_k : int, end_k : int) :
    from collections import defaultdict

    img = cv.imread(img_path)
    sil_arr = np.array([])
    
    img_flat = img.reshape(-1, 3)

    for k in range (start_k, end_k + 1) :
        print(k)
        res = pysignals.seq.k_means(img, k, 0.01)
        res_flat = res.reshape(-1, 3)

        # Create a map / dictionary (Pixel -> array of pixels) [Prototype / centroid -> all pixel belonging to that]
        cluster_map = defaultdict(list)
    
        # Create the mapping: cluster index -> list of pixels in that cluster
        for pixel, cluster in zip(img_flat, res_flat):
            cluster_map[tuple(cluster)].append(pixel)
        
        # Silouette computation
        

    # Plot
    # plt.plot(np.arange(start_k, end_k + 1), sil_arr, '-o')
    # plt.title('Silouette method')
    # plt.xlabel('Number of clusters k')
    # plt.ylabel('Average silouette width')
    # plt.savefig('../data/stats/silouette.jpg')


