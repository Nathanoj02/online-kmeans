import sys
import os
import cv2 as cv

from algorithms import k_means

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "python"))

import pysignals

if __name__ == '__main__':

    img = cv.imread('../data/car.jpg')

    dst = pysignals.seq.k_means(img, 5, 0.1)
    # res = k_means(img, 5, 5)

    cv.imwrite(f'../data/kmeans_cpp.jpg', dst)

    # dst = pysignals.par.k_means(img, 5, 0.1)

    # for k in range(1, 11) :
    #     img = cv.imread('../data/car.jpg')

    #     res = k_means(img, k, 1)

    #     cv.imwrite(f'../data/kmeans_{k}.jpg', res)
    
