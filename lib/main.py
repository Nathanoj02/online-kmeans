import cv2 as cv

from algorithms import k_means, k_means_cpp, k_means_cuda
from metrics import elbow_method, silouette_method

if __name__ == '__main__':

    # elbow_method('../data/car.jpg', 1, 15)
    silouette_method('../data/car.jpg', 2, 10)


    # img = cv.imread('../data/car.jpg')

    # res = pysignals.par.k_means(img, 5, 1)
    # res = k_means(img, 5, 5)

    # cv.imwrite(f'../data/kmeans_cuda.jpg', res)

    # res = pysignals.seq.k_means(img, 5, 0.1)

    # for k in range(1, 11) :
    #     img = cv.imread('../data/car.jpg')

        # res = pysignals.seq.k_means(img, k, 0.01)

    #     cv.imwrite(f'../data/kmeans_{k}.jpg', res)
    
