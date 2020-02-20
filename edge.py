import numpy as np
import math
import cv2
from scipy.stats import multivariate_normal
from scipy import ndimage

##I called this one mein instead of my because I thought it would spice things up a little
## but this is where I control the flow of the program. 
def meinEdgeFilter(img0, sigma):	

	threshold = 40
	
	gauss = convolve(img0,generateGaussianKernel(sigma))
	cv2.imwrite ('gauss.png', gauss)
	
	kernX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	kernY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	
	sobelX = convolve(gauss, kernX)
	sobelY = convolve(gauss, kernY)
	
	cv2.imwrite("sobelx.png", sobelX)
	cv2.imwrite("sobely.png", sobelY)
	
	mag = np.hypot(sobelX, sobelY)
	mag = mag / mag.max() * 255
	cv2.imwrite("magnitude.png", mag)
	
	theta = np.arctan2(sobelX, sobelY)
	
	for i in range(0, img0.shape[0]):
		for j in range(0, img0.shape[1]):
			if(mag[i][j] > threshold):
				mag[i][j] = 255
			else:
				mag[i][j] = 0
	
	return mag

#non-maxima suppression, this doesn't do anything or work because I was confused by it and ran out of time
def NMS(img, D):
   print("NMS")
	
#generates a gaussian kernal using sigma as a seed for the kernel size  
def generateGaussianKernel(sigma):
	size = 2 * math.ceil(3*sigma)+1
	x, y = np.mgrid[-size:size+1, -size:size+1]
	normal = 1 / (2.0 * np.pi * sigma**2)
	g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
	return g
	
#my implementation of the convolution function
def convolve(image, kernel):
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
	for y in range(pad, iH + pad):
		for x in range(pad, iW + pad):
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			k = (roi * kernel).sum()
			output[y - pad, x - pad] = k
	# return the output image
	return output
	
output = meinEdgeFilter(cv2.imread('cat2.png',0) , 1.4)

cv2.imwrite ('output.png', output)
cv2.imshow('output',output)
cv2.waitKey(0)
cv2.destroyAllWindows()