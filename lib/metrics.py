import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import time
import os

from algorithms import k_means, k_means_cpp, k_means_cuda, k_means_cuda_shared_mem, k_means_scikit

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "python"))

import pysignals


def _elbow_method (img : np.ndarray, clustered_img : np.ndarray, k : int) :
    # Calculate sum of squared error
    error = np.sum((img - clustered_img) ** 2)
    return error


def elbow_method (img : np.ndarray, start_k : int, end_k : int) :
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


def silhouette_method (img : np.ndarray, start_k : int, end_k : int) :
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


def metrics (img : np.ndarray, start_k : int, end_k : int) :
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


def _execution_time_step (algorithm : callable, img : np.ndarray, k : int, stab_error : int, tries : int = 10) -> list[float] :
    times = []

    for i in range(tries) :
        dusk = time.time()
        algorithm(img, k, stab_error)
        dawn = time.time()
        times.append(dawn - dusk)
    
    return times


def execution_time_avg (img : np.ndarray, k : int, stab_error : int, tries : int = 10) -> tuple :
    times = _execution_time_step(k_means_scikit, img, k, stab_error, tries)
    py_time_avg = np.mean(times)
    py_time_std = np.std(times)
    
    times = _execution_time_step(k_means_cpp, img, k, stab_error, tries)
    cpp_time_avg = np.mean(times)
    cpp_time_std = np.std(times)
    
    # Do once to "warm up" GPU
    k_means_cuda(img, k, stab_error)
    
    times = _execution_time_step(k_means_cuda, img, k, stab_error, tries)
    cuda_time_avg = np.mean(times)
    cuda_time_std = np.std(times)

    times = _execution_time_step(k_means_cuda_shared_mem, img, k, stab_error, tries)
    cuda_shared_time_avg = np.mean(times)
    cuda_shared_time_std = np.std(times)

    times = []
    dev = pysignals.par.init_k_means(img.shape[0], img.shape[1], k)

    for i in range(tries) :
        dusk = time.time()
        pysignals.par.k_means(img, k, stab_error, 300, dev, True)
        dawn = time.time()
        times.append(dawn - dusk)
    
    pysignals.par.deinit_k_means(dev)

    cuda_video_time_avg = np.mean(times)
    cuda_video_time_std = np.std(times)

    return py_time_avg, py_time_std, \
        cpp_time_avg, cpp_time_std, \
        cuda_time_avg, cuda_time_std, \
        cuda_shared_time_avg, cuda_shared_time_std, \
        cuda_video_time_avg, cuda_video_time_std


def plot_execution_times (img : np.ndarray, k : int, stab_error : int, tries : int = 10) :

    py_time_avg, _, cpp_time_avg, _, cuda_time_avg, _, cuda_shared_time_avg, _, cuda_video_time_avg, _ = execution_time_avg (img, k, stab_error, tries)

    labels = ['Scikit-learn', 'C++', 'CUDA single image', 'CUDA shared memory', 'CUDA video']
    times = [py_time_avg, cpp_time_avg, cuda_time_avg, cuda_shared_time_avg, cuda_video_time_avg]
    times_strings = ["{:.4f}".format(t) for t in times]

    plt.bar(labels, times)
    plt.xticks(rotation=60)
    plt.gcf().set_constrained_layout(True)

    # Add labels
    for i in range (len(labels)):
        plt.text(i, times[i], times_strings[i], ha = 'center')

    plt.title('Average k-means execution time')
    plt.xlabel('Method')
    plt.ylabel('Execution time [s]')
    plt.savefig('../data/stats/times.jpg')

