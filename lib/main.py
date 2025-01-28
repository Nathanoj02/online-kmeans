import cv2 as cv

from algorithms import k_means, k_means_cpp, k_means_cuda
from metrics import elbow_method, silhouette_method

if __name__ == '__main__':

    # elbow_method('../data/car.jpg', 1, 15)
    # silouette_method('../data/car.jpg', 2, 10)


    img = cv.imread('../data/car.jpg')
    res = k_means_cuda(img, 2, 0.5)
    cv.imwrite(f'../data/kmeans_cuda.jpg', res)

    # res = k_means_cpp(img, 2, 0.5)
    # cv.imwrite(f'../data/kmeans_cpp.jpg', res)
    
