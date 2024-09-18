import sys
import os
# import cv2 as cv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "python"))

import pysignals

if __name__ == '__main__':
    pysignals.seq.foo()
    # img = cv.imread('prova.jpg')
    
