import sys
import os
import cv2 as cv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from build.module_name import *

if __name__ == '__main__':
    foo()
    img = cv.imread('prova.jpg')
    
