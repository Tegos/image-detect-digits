import cv2
from find_obj import filter_matches, explore_match

img1 = cv2.imread('../c/box.png', 0)  # queryImage
img2 = cv2.imread('../c/box_in_scene.png', 0)  # trainImage

# Initiate SIFT detector
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # , crossCheck=True)

matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
p1, p2, kp_pairs = filter_matches(kp1, kp2, matches)
explore_match('find_obj', img1, img2, kp_pairs)  # cv2 shows image

cv2.waitKey()
cv2.destroyAllWindows()
