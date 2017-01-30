import os

import cv2

from config import digitheight


def get_line_param(p1, p2):
    x1 = float(p1[0])
    y1 = float(p1[1])

    x2 = float(p2[0])
    y2 = float(p2[1])

    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    return k, b


# draw line throw to point to full screen
def draw_full_line(point1, point2, img):
    k, b = get_line_param(point1, point2)
    height, width, ch = img.shape

    x1 = 0
    y1 = k * x1 + b

    x2 = width
    y2 = k * x2 + b

    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    cv2.line(img, p1, p2, (0, 255, 255), 2)


def getDigitFromImage(im):
    from sklearn.externals import joblib
    from skimage.feature import hog
    import numpy as np

    # Load the classifier
    digits_cls = os.path.join(os.path.dirname(__file__), 'res/digits_cls.pkl')
    # clf, pp = joblib.load('res/digits_cls.pkl')
    clf, pp = joblib.load(digits_cls)

    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = im

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 3, 5, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    # ctrs, hier = cv2.findContours(im_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ret, im_th_inv = cv2.threshold(im_gray, 10, 50, cv2.THRESH_BINARY_INV)

    used_img = im_th_inv

    cv2.imshow('used_img', used_img)

    ctrs, hier = cv2.findContours(used_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for ctr in ctrs:
        rect = cv2.boundingRect(ctr)
        x, y, w, h = cv2.boundingRect(ctr)
        if h > w and h > (digitheight * 4) / 5:
            # Draw the rectangles
            cv2.rectangle(used_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 1)
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
            roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
            nbr = clf.predict(roi_hog_fd)
            cv2.putText(used_img, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            print 'main digit: ' + str(int(nbr[0]))
    cv2.imshow('digit detect', used_img)


def get_line_coord_perpendicular(p1, p2, dist, first=True):
    x1 = float(p1[0])
    y1 = float(p1[1])

    x2 = float(p2[0])
    y2 = float(p2[1])

    if first:
        x = x1
        y = y1

    else:
        x = x2
        y = y2

    k, b = get_line_param(p1, p2)

    y_new = int(y + dist)
    x_new = int(k * (y - y_new) + x)
    return x_new, y_new


def getRandomString(size=6):
    import random
    import string
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated
