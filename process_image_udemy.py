import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import differint.differint as df
from skimage.morphology import skeletonize, thin
import cv2

pic = Image.open('database/101_1.tif')
type(pic)

pic_arr = np.asarray(pic)
type(pic_arr)


pic_arr.shape
plt.imshow(pic_arr, cmap='gray')
plt.show()


import cv2
img = cv2.imread('101_1.tif')
type(img)

img.shape
plt.imshow(img)
plt.show()

# MATPLOTLIB --> RGB
# OPENCV --> BGR

fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)
plt.show()

img_gray = cv2.imread('101_1.tif', cv2.IMREAD_GRAYSCALE)
# without 3rd dimension

# grayscale does't work?
img_gray.shape
plt.imshow(img_gray)
plt.show()


# resize image
new_img = cv2.resize(fix_img, (224, 224))
plt.imshow(new_img)
plt.show()

# picture 50% smaller
w_ratio = 0.5
h_ratio = 0.5


new_img = cv2.resize(fix_img, (0, 0), fix_img,w_ratio,h_ratio)
new_img.shape


while True:
    cv2.imshow('Fingerprint', img)

    # IF we've waited at leat 1 ms AND w've passed ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()


# thresholding image to consist only two values => black and white

# my threshold tpically half of the 255

img.max()
img.min()

# żeby użyc OTSU obraz musi być  wczytany w grayscale
img_gray = cv2.imread('101_1.tif', cv2.IMREAD_GRAYSCALE)
img_gray.shape
plt.imshow(img_gray)
plt.show()

# poziom tresholdu połowa 255 zazwyczaj ale mozna eksperymentowac (do Otsu tez?)
ret, thresh1 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_OTSU) # THRESH_BINARY_INV => wracamy do poprzedniego
print(ret)
                                                               # THRESH_OTSU  => uzywa algorytmu otsu, musi byc w grayscale
plt.imshow(thresh1, cmap='gray') # bez cmpay='gray' wczyta fioletowo żółty
plt.show()

# block_size (3 ,5, 7, 11 zazwyczaj), constant mean albo weighted_mean => strzela
th2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 8)
plt.imshow(th2, cmap='gray')
plt.show()


blended = cv2.addWeighted(src1=thresh1, alpha=0.6, src2=th2, beta=0.4, gamma=0)
plt.imshow(blended, cmap='gray')
plt.show()


# often blurring or smoothing (zamazywanie, wygładzanie) is combined with edge detection



############################################################


def load_img():
    img = cv2.imread('101_1.tif').astype(np.float32) / 255
    return img


def show_img(image, cmap=None):
    if cmap:
        plt.imshow(image, cmap)
        plt.show()
    else:
        plt.imshow(image)
        plt.show()

fingerprint = load_img()
show_img(fingerprint)

image_array = load_img()

print(image_array)


kernel = np.matrix([[1, 2, 1], [2, 4, 2], [1, 2, 1]])


# ddepth parameter = - 1
destination = cv2.filter2D(fingerprint, -1, kernel)
show_img(destination.astype(np.uint8))  # np.uint8 to dismiss warning


x = gaussian_filter(kernel, sigma=-2)
print(x)

#pochodna ulamkowa rzedu v
v = 0.1
new_kernel = df.RL(v, kernel, 1, 5, 3)
new_destination = cv2.filter2D(fingerprint, -1, new_kernel)
show_img(new_destination.astype(np.uint8))


# test pochodnej ulamkowej funkcji w punkcie (porownac z macierza)
def f(x):
   return x


DF = df.RLpoint(0.5, f)
print(DF)



blurred_fingerprint = cv2.GaussianBlur(fingerprint, (3, 3), 10)
show_img(blurred_fingerprint)


show_img(img)


# Edge detection

# Sobel operator
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
show_img(sobelx.astype(np.uint8))

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
show_img(sobely.astype(np.uint8))


# Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)
show_img(laplacian)


# image histograms  OPENCV BGR  (0,255) wlasciwie  (i tak nic nie bedzie widac na histogramie)
hist_values = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(hist_values)
plt.show()






# Harris Corner Detection

# Cheesboard
img = cv2.imread('Chess_board.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
type(img)
img.shape
show_img(img)
plt.imshow(img_gray, cmap='gray')
plt.show()

gray = np.float32(img_gray)
# blocksize= neigbourhood size, ksize=sobol operator kernel size, harrisdetectorfree paramater
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)


img[dst > 0.01 * dst.max()] = [255, 0, 0] # if 1% of max value that mark as red
show_img(img)

plt.imshow(img_gray)

plt.show()


# Fingerprint Harris
img = cv2.imread('101_1.tif')
img_gray = cv2.imread('101_1.tif',0) # grayscale
show_img(img)
show_img(img_gray,cmap='gray')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img)
plt.show()


plt.imshow(img, cmap='gray')
plt.show()

# wczytanie w szarości
gray = np.float32(img_gray)

#
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img_gray, (5,5), 0)
plt.imshow(blur, cmap='gray')
plt.show()

# global thresholding
ret1, th1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
show_img(th1,cmap='gray')

# Otsu's thresholding
ret2,th2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
show_img(th2, cmap='gray')


ret, img0 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
show_img(img0,cmap='gray')

# dlaczego polaczenie tresh binary i otsu?
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3, th3 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
show_img(th3,cmap='gray')

# szkieletyzacja
keleton = skeletonize(th2) # => nie mozna wykonac szkieletyzacji bo nie sa same 0 i 1
# Normalize to 0 and 1 range
th2[th2 == 255] = 1
skeleton = skeletonize(th2)
show_img(skeleton, cmap='gray')
# to u niego w kjanko -> po co to
#skeleton = numpy.array(skeleton, dtype=numpy.uint8)
#skeleton = removedot(skeleton)



# blocksize= neigbourhood size, ksize=sobol operator kernel size, harrisdetectorfree paramater
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)


img[dst > 0.01 * dst.max()] = [255, 0, 0] # if 1% of max value that mark as red
show_img(img)
plt.imshow(img_gray)
plt.show()

# Shi Tomasi
img = cv2.imread('101_1.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# number of corner (-1) for all corners , quality parametr , good 0.01, distance? good = 10
corners = cv2.goodFeaturesToTrack(img_gray, 10**4, 0.01, 10)
print(corners)

# to integer becouse want to draw circles
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x,y), 3, (255, 0, 0), -1)

plt.imshow(img)
plt.show()


# Edge detection
# Canny algorithm

# Algorithm


#1. Apply Gaussian filter to smooth the image and remove noise
#
img = cv2.imread('101_1.tif')
edges = cv2.Canny(image=img, threshold1=0, threshold2=127)
show_img(edges)

med_val = np.median(img)
print(med_val)

# lower threshold to either 0 or 70% of the median value whichever is grater
lower = int(max(0, 0.7*med_val))
# upper threshold to either 130$  of the median or the max 255, whichever is smaller
upper = int(min(255, 1.3*med_val))

edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper)
show_img(edges)

blurred_img = cv2.blur(img,ksize=(5,5))
edges = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper)
show_img(edges)


# Feature matching

#1 Brute-Force matching with ORB Descriptors
#2 Brute-Force matching with SIFT Descriptors and Ratio Test
#3 Flann based matcher
finger_1 = cv2.imread('101_1.tif')
finger_2 = cv2.imread('101_2.tif')
show_img(finger_1)
show_img(finger_2)


# create detector
orb = cv2.ORB_create()
# find key pointss and descritpors

# image mask[, descriptors[, useProvidedKeypoints]] -> keypoints descriptors
kp1, des1 = orb.detectAndCompute(finger_1, None)
kp2, des2 = orb.detectAndCompute(finger_2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  #(default paramters)
matches = bf.match(des1, des2)


single_match = matches[0]
single_match.distance  # good match = small distance

# sort them by distance
matches - sorted(matches, key=lambda x:x.distance)
len(matches)    # 190

final_matches = cv2.drawMatches(finger_1, kp1, finger_2, kp2, matches[:25], None, flags=2)
show_img(final_matches)  # don't good !!!!!!


# SIFT (good for image on different scale)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(finger_1, None)
kp2, des2 = sift.detectAndCompute(finger_2, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2) # two best matches


# ratio test - sprawdzamy czy matche sa dobre
good = []

for match1, match2 in matches:
    # IF MATCH 1 DISTANCE is LESS THAN 75% of MATCH DISTANCE
    # THE DESCRIPTOR WAS A GOOD MATCH, LETS KEEP IT!
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

len(good)
len(matches)

sift_matches = cv2.drawMatchesKnn(finger_1,kp1, finger_2, kp2, good, None, flags=2)




# convolutional network
from keras.datasets import mnist


















