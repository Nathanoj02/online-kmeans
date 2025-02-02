import cv2 as cv

from algorithms import k_means, k_means_cpp, k_means_cuda, k_means_cuda_shared_mem, k_means_scikit, k_means_video, k_means_live
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


def test_video () :
    cap = cv.VideoCapture('../data/walking.mp4')
    k_means_video(cap, 3, 0.5, save_path = '../data/video_res.mp4')


def test_live () :
    k_means_live(3, 0.5)


if __name__ == '__main__':
    test_live()
    
