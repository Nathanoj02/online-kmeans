import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import time

from algorithms import k_means_cpp


def _elbow_method (img : np.ndarray, clustered_img : np.ndarray, k : int) :
    # Calculate sum of squared error
    error = np.sum((img - clustered_img) ** 2)
    return error


def elbow_method (img_path : str, start_k : int, end_k : int) :
    img = cv.imread(img_path)
    error_arr = np.array([])

    for k in range (start_k, end_k + 1) :
        res = k_means_cpp(img, k, 0.01)
        error = _elbow_method(img, res, k)
        error_arr = np.append(error_arr, error)

    # Plot
    plt.plot(np.arange(start_k, end_k + 1), error_arr, '-o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Total within sum of squares')
    plt.savefig('../data/stats/elbow.jpg')


def _silhouette_method (img : np.ndarray, clustered_img : np.ndarray, k : int) :
    img_flat = img.reshape(-1, 3)
    clustered_img_flat = clustered_img.reshape(-1, 3)

    # Preprocess clustered image
    _, labels = np.unique(clustered_img_flat, axis=0, return_inverse=True)

    dusk = time.time()
    silhouette_avg = silhouette_score(img_flat, labels, metric='euclidean')
    dawn = time.time()
    print(silhouette_avg)
    print(f'k = {k}\nTime spent = {(dawn - dusk) / 60} min', end='\n\n')

    return silhouette_avg


def silhouette_method (img_path : str, start_k : int, end_k : int) :
    img = cv.imread(img_path)
    sil_arr = []
    
    for k in range (start_k, end_k + 1) :
        print(k)
        sys.stdout.flush()
        res = k_means_cpp(img, k, 0.01)
        
        silhouette_avg = _silhouette_method(img, res, k)
        sil_arr.append(silhouette_avg)

    # Plot
    plt.plot(np.arange(start_k, end_k + 1), sil_arr, '-o')
    plt.title('Silhouette method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Average silouette width')
    plt.savefig('../data/stats/silhouette.jpg')


def metrics (img_path : str, start_k : int, end_k : int) :
    img = cv.imread(img_path)
    error_arr = []
    sil_arr = []

    for k in range(start_k, end_k + 1) :
        res = k_means_cpp(img, k, 0.01)

        error = _elbow_method(img, res, k)
        silhouette_avg = _silhouette_method(img, res, k) if k >= 2 else 0

        error_arr.append(error)
        sil_arr.append(silhouette_avg)
    
    # Plot elbow
    plt.plot(np.arange(start_k, end_k + 1), error_arr, '-o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Total within sum of squares')
    plt.savefig('../data/stats/elbow.jpg')

    # Plot silhouette
    plt.plot(np.arange(start_k, end_k + 1), sil_arr, '-o')
    plt.title('Silhouette method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Average silouette width')
    plt.savefig('../data/stats/silhouette.jpg')