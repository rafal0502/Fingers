import cv2
from matplotlib import pyplot as plt
import numpy as np


img = cv2.imread("./database/101_1.tif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg', img)


img2 = cv2.imread("./database/102_6.tif")
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift2 = cv2.xfeatures2d.SIFT_create()
kp2 = sift.detect(gray2, None)
img2 = cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints2.jpg', img2)


img3 = cv2.imread("./database/101_3.tif")
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
sift3 = cv2.xfeatures2d.SIFT_create()
kp3 = sift.detect(gray3, None)
img3 = cv2.drawKeypoints(gray3, kp3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints3.jpg', img3)


img4 = cv2.imread("./database/101_4.tif")
gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
sift4 = cv2.xfeatures2d.SIFT_create()
kp4 = sift.detect(gray4, None)
img4 = cv2.drawKeypoints(gray4, kp4, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints4.jpg', img4)


# Now to calculate the descriptor, OpenCV provides two methods.
kp, des = sift.compute(gray, kp)
kp2, des2 = sift.compute(gray2, kp2)
kp3, des3 = sift.compute(gray3, kp3)
kp4, des4 = sift.compute(gray4, kp4)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)   #NORM_HAMING

# match descriptors
matches = bf.match(des, des2)


# sort them in the order of their distance
matches = sorted(matches, key=lambda x:x.distance)


# Draw first 10 matches
img_img2 = cv2.drawMatches(img, kp, img2, kp2, matches[:10], None, flags=2)
plt.imshow(img_img2)
plt.show()


from siftmatch import match_template
match_template("./database/101_1.tif", "./database/102_6.tif", 5, 2)


