import cv2
import imutils as imutils
import numpy as np
from matplotlib import pyplot as plt
from functions import *
from config import *

image_path = '../res/555.png'
image_template = '../res/template.png'
image_bound_open = '../res/bound_open.png'
image_bound_close = '../res/bound_close.png'

img_rgb = cv2.imread(image_path)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(image_template, 0)

bound_bracket_open = cv2.imread(image_bound_open, 0)
bound_bracket_close = cv2.imread(image_bound_close, 0)

img_original = img_rgb.copy()
threshold = 0.7

w, h = template.shape[::-1]
searches = [bound_bracket_close, bound_bracket_close]

for r_angle in range(0, 360, 5):
    for search in searches:
        rotated_bracket = imutils.rotate_bound(search, r_angle)
        rotated_bracket = rotate(search, r_angle)

        cv2.imshow('rotated_bracket', rotated_bracket)
        # cv2.waitKey(0)
        res = cv2.matchTemplate(img_gray, rotated_bracket, cv2.TM_CCOEFF_NORMED)

        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            bound_1 = pt
            bound_2 = (pt[0] + w, pt[1] + h)
            cv2.rectangle(img_rgb, bound_1, bound_2, (0, 0, 255), 1)

            # break

cv2.namedWindow('Detected', cv2.CV_WINDOW_AUTOSIZE)
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_result/' + getRandomString() + 'res.png', img_rgb)
