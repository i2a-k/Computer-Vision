import numpy as np
import cv2 as cv
from skimage import color
import skimage.feature as ft

class Comparator:
    def __init__(self, data, query):
        self.data = data
        self.query = query
    
    def colorHistogram(self):
        colors = [0, 1, 2] #B, G, R
        inputColorHist = []
        score = np.inf
        for i in colors:
            inputColorHist.append(cv.calcHist(self.query,[colors[i]],None,[256],[0,256]))
        for j in self.data:
            T = 0
            for k in colors:
                temp = cv.calcHist(j,[colors[k]],None,[256],[0,256])
                T += cv.compareHist(inputColorHist[k], temp, cv.HISTCMP_CHISQR)
            if(T <= score):
                score = T
                outputImage = j
        return outputImage
        
    def lbp(self):
        score = np.inf
        img = color.rgb2gray(self.query)
        patterns = ft.local_binary_pattern(img, 8, 2)
        n_bins = int(patterns.max() + 1)
        inputLBPHist, _ = np.histogram(patterns, bins=n_bins, range=(0, n_bins), density=True)
        for j in self.data:
            T = 0
            grayImg = color.rgb2gray(j)
            lbpPatterns = ft.local_binary_pattern(grayImg, 8, 2)
            n_bins = int(lbpPatterns.max() + 1)
            lbpHist, _ = np.histogram(lbpPatterns, bins=n_bins, range=(0, n_bins), density=True)
            T = 0.5*np.sum((inputLBPHist-lbpHist)**2/(inputLBPHist+lbpHist+1e-6)) #Chi-Square
            if(T <= score):
                score = T
                outputImage = j
        return outputImage        