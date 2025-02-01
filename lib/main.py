import cv2 as cv

from algorithms import k_means, k_means_cpp, k_means_cuda, k_means_cuda_shared_mem
from metrics import elbow_method, silhouette_method, metrics, plot_execution_times


def test_exec_times () :
    img = cv.imread('../data/car.jpg')
    plot_execution_times(img, 5, 0.5)


def test_metrics () :
    img = cv.imread('../data/car.jpg')
    metrics(img, 1, 10)


def test_algorithms () :
    img = cv.imread('../data/car.jpg')
    
    res = k_means(img, 5, 0.5)
    cv.imwrite(f'../data/kmeans_python.jpg', res)

    res = k_means_cpp(img, 5, 0.5)
    cv.imwrite(f'../data/kmeans_cpp.jpg', res)

    res = k_means_cuda(img, 5, 0.5)
    cv.imwrite(f'../data/kmeans_cuda.jpg', res)

    res = k_means_cuda_shared_mem(img, 5, 0.5)
    cv.imwrite(f'../data/kmeans_cuda_shared.jpg', res)


if __name__ == '__main__':
    # test_exec_times()
    img = cv.imread('../data/car.jpg')
    res = k_means_cuda_shared_mem(img, 5, 0.5)
    cv.imwrite(f'../data/kmeans_cuda.jpg', res)
    
