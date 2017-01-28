import cv2
import numpy as np
from matplotlib import pyplot as plt
from functions import *

image_path = '../res/555.png'
# image_template = '../res/search.png'
image_template = '../res/template.png'

img_rgb = cv2.imread(image_path)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(image_template, 0)

w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bound_1 = pt
    bound_2 = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, bound_1, bound_2, (0, 0, 255), 1)
    # draw diagonal
    line_coord_x = (bound_1[0], bound_1[1] + h)
    line_coord_y = (bound_2[0], bound_1[1])
    draw_full_line(line_coord_x, line_coord_y, img_rgb)

    cv2.line(img_rgb, line_coord_x, line_coord_y, (0, 255, 0), 3)

# cv2.imshow("Detected", edges)

# cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('res.png', img_rgb)
