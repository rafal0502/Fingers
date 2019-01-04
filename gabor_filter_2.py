from skimage.filters import gabor
from skimage import data, io
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("101_1.tif")
filt_real, filt_img = gabor(image, frequency=0.6)
plt.figure()
io.imshow(filt_real)
io.show()

