import cv2 as cv

from algorithms import k_means, k_means_cpp, k_means_cuda, k_means_cuda_shared_mem, k_means_scikit
from metrics import metrics, plot_execution_times


def test_exec_times () :
    img = cv.imread('../data/video_frame.jpg')
    plot_execution_times(img, 3, 0.5)


def test_metrics () :
    img = cv.imread('../data/video_frame.jpg')
    metrics(img, 1, 10)


def test_algorithms () :
    img = cv.imread('../data/car.jpg')
    
    res = k_means(img, 3, 0.5)
    cv.imwrite(f'../data/kmeans_python.jpg', res)

    res = k_means_scikit(img, 3, 0.5)
    cv.imwrite(f'../data/kmeans_scikit.jpg', res)

    res = k_means_cpp(img, 3, 0.5)
    cv.imwrite(f'../data/kmeans_cpp.jpg', res)

    res = k_means_cuda(img, 3, 0.5)
    cv.imwrite(f'../data/kmeans_cuda.jpg', res)

    res = k_means_cuda_shared_mem(img, 3, 0.5)
    cv.imwrite(f'../data/kmeans_cuda_shared.jpg', res)



if __name__ == '__main__':
    img = cv.imread('../data/video_frame.jpg')
    res = k_means_cuda_shared_mem(img, 3, 0.5)
    cv.imwrite(f'../data/kmeans_cuda_shared.jpg', res)
    
