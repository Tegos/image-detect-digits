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
image_bound_close = '../res/909.png'

img_rgb = cv2.imread(image_path)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(image_template, 0)

bound_bracket_open = cv2.imread(image_bound_open, 0)
bound_bracket_close = cv2.imread(image_bound_close, 0)

img_original = img_rgb.copy()
threshold = 0.7

template = cv2.imread('../res/bracket_middle.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 20, 100)
(tH, tW) = template.shape[:2]
# template_roi = template[5:tH - 5, 5:tW - 5]
cv2.imshow("Template", template)

searches = [template]

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", image)

for r_angle in range(0, 360, 5):
    print r_angle
    for search in searches:
        rotated_bracket = imutils.rotate_bound(search, r_angle)

        found = None
        # loop over the scales of the image
        for scale in np.linspace(0.5, 1.0, 10)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 20, 100)

            result = cv2.matchTemplate(edged, rotated_bracket, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # check to see if the iteration should be visualized
            if 0:
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)

            # if we have found a new maximum correlation value, then ipdate
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # draw a bounding box around the detected result and display the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # rect = cv2.minAreaRect(np.array([(startX, startY), (endX, endY)],
        #                                 dtype=np.int32
        #                                 ))
        # box = cv2.cv.BoxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)

# cv2.namedWindow('Detected', cv2.CV_WINDOW_AUTOSIZE)
# cv2.imshow('Detected', img_rgb)
# cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image_result/' + getRandomString() + 'res.png', img_rgb)
