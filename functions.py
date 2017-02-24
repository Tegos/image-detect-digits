import os

import Image
import numpy as np
import cv2

from config import *


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
    height, width = img.shape

    x1 = 0
    y1 = k * x1 + b

    x2 = width
    y2 = k * x2 + b

    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    if DEBUG:
        cv2.line(img, p1, p2, (0, 255, 255), 2)
    return p1, p2


def getDigitFromImage(im):
    from sklearn.externals import joblib
    from skimage.feature import hog

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
        if h > w and h > (digit_height * 4) / 5:
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


def getDigitFromImageNew(im):
    from sklearn.externals import joblib
    from skimage.feature import hog

    # Load the classifier
    digits_cls = os.path.join(os.path.dirname(__file__), 'res/digits_cls.pkl')
    # clf, pp = joblib.load('res/digits_cls.pkl')
    clf, pp = joblib.load(digits_cls)

    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = im
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for ctr in ctrs:
        rect = cv2.boundingRect(ctr)
        x, y, w, h = cv2.boundingRect(ctr)
        # if h > w and h > (digitheight * 4) / 5:
        if h > (digit_height * 4) / 5:
            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 1)
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_gray[pt1:pt1 + leng, pt2:pt2 + leng]
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
            roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
            nbr = clf.predict(roi_hog_fd)
            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 1)
            print 'main digit: ' + str(int(nbr[0]))

    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    cv2.imshow('im', im)
    cv2.waitKey(0)


def getDigitFromImageSimple(im):
    from sklearn.externals import joblib
    from skimage.feature import hog

    # Load the classifier
    digits_cls = os.path.join(os.path.dirname(__file__), 'res/digits_cls.pkl')
    # clf, pp = joblib.load('res/digits_cls.pkl')
    clf, pp = joblib.load(digits_cls)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    roi = im_gray
    # Threshold the image
    ret, im_th = cv2.threshold(roi, 190, 250, cv2.THRESH_BINARY_INV)
    # roi = im_th

    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
    # cv2.imshow('roi', im_th)
    # cv2.waitKey(0)

    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
    nbr = clf.predict(roi_hog_fd)
    # cv2.putText(used_img, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    print 'main digit: ' + str(int(nbr[0]))
    print 'predict: '
    print nbr
    return int(nbr[0])


def getDigitImage(im):
    print im


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


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.namedWindow('Features', cv2.WINDOW_NORMAL)
    cv2.imshow('Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


def cropImage(image_source, point_1, point_2):
    x1, y1 = int(point_1[0]), int(point_1[1])
    x2, y2 = int(point_2[0]), int(point_2[1])
    cropped = image_source[y1:y2, x1:x2]
    return cropped


def findCoordStartEndBracket(img, img_bracket_middle, img_start_bracket, img_end_bracket, point_1, point_2):
    w_img, h_img = img.shape[::-1]
    w, h = img_bracket_middle.shape[::-1]
    result_1 = None
    result_2 = None
    k_size = k_size_bound_on_bracket

    # region find start bracket
    add_y_min = point_1[1] - h
    add_y_max = point_1[1] + h

    start_x = point_1[0]
    add_x = point_1[0]
    counter = 0
    stop = False
    while add_x > 0 and not stop:
        new_point_1 = (add_x - w, add_y_min)
        new_point_2 = (start_x, add_y_max)

        x1, y1 = int(new_point_1[0]), int(new_point_1[1])
        x2, y2 = int(new_point_2[0]), int(new_point_2[1])

        cropped = img[y1:y2, x1:x2]

        find_pos = MultiScaleSearchTemplate(cropped, img_start_bracket)

        stop = find_pos[0]
        if stop:
            if DEBUG:
                print find_pos[0], find_pos[1], counter
                cv2.rectangle(img, new_point_1, new_point_2, (0, 0, 255), 2)
            result_1 = (new_point_1[0], new_point_1[1] - h * k_size)

        add_x -= w / 2
        counter += 1
    # endregion

    # find end bracket
    start_x = point_2[0]
    add_x = point_2[0]
    counter = 0
    stop = False
    # print point_2

    # cv2.namedWindow('img_end_bracket', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('img_start_bracket', cv2.WINDOW_NORMAL)
    # cv2.imshow('img_end_bracket', img_end_bracket)
    # cv2.imshow('img_start_bracket', cropped)
    # cv2.waitKey(0)
    while add_x <= w_img and not stop:
        new_point_1 = (start_x, add_y_min)
        new_point_2 = (add_x + w, add_y_max)

        x1, y1 = int(start_x), int(add_y_min)
        x2, y2 = int(add_x + w), int(add_y_max)

        # if isInt(x1)

        cropped = img[y1:y2, x1:x2]

        # cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
        # cv2.imshow('cropped', cropped)
        # cv2.waitKey(0)

        find_pos = MultiScaleSearchTemplate(cropped, img_end_bracket)

        stop = find_pos[0]
        if stop:
            if DEBUG:
                print find_pos[0], find_pos[1], counter
                cv2.rectangle(img, new_point_1, new_point_2, (0, 0, 255), 2)
            result_2 = (new_point_2[0], new_point_2[1])

        add_x += w / 2
        counter += 1

    return result_1, result_2


def MultiScaleSearchTemplate(img, template, threshold=0.75):
    import imutils
    location = False
    found = False
    w_img, h_img = img.shape[::-1]

    for scale in np.linspace(0.6, 1.7, 30)[::-1]:
        # print scale, found
        if found:
            break
        resize_template = imutils.resize(template, width=int(template.shape[1] * scale))

        # cv2.imshow('img_edges', img_edges)
        # cv2.imshow('template_edges', template_edges)
        # cv2.waitKey(0)
        w, h = resize_template.shape[::-1]
        if w_img < w or h_img < h:
            continue

        thresh_template = cv2.threshold(resize_template, 0, 255,
                                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        thresh_img = cv2.threshold(img, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
        # cv2.imshow('thresh', thresh_template)
        # cv2.waitKey(0)

        # result = cv2.matchTemplate(img, resize_template, cv2.TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(thresh_img, thresh_template, cv2.TM_CCOEFF_NORMED)

        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            bound_1 = pt
            bound_2 = (pt[0] + w, pt[1] + h)
            if DEBUG:
                cv2.rectangle(img, bound_1, bound_2, (0, 0, 255), 2)
            location = (bound_1, bound_2)
            # print location
            found = True
            break

    return found, location


def isImage(image):
    w, h = image.shape[::-1]
    if w > 0 and h > 0:
        return True
    return False


def secondsToStringTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print "%d:%02d:%02d" % (h, m, s)


def isInt(obj):
    res = isinstance(obj, (int, long))
    return res


def EdgeDetect(file_name, thresh_min, thresh_max):
    image = cv2.imread(file_name)
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.GaussianBlur(im_bw, (1, 1), 0)
    # image = AutCanny(gray)

    (thresh, im_bw) = cv2.threshold(im_bw, thresh_min, thresh_max, 0)
    # cv2.imwrite(file_name + '_bw.png', im_bw)

    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    # cv2.imwrite(file_name + '_ctn.png', image)

    for i in range(0, len(contours)):
        cnt = contours[i]
        # mask = np.zeros(im2.shape,np.uint8)
        # cv2.drawContours(mask,[cnt],0,255,-1)
        x, y, w, h = cv2.boundingRect(cnt)
        if h > w and h > (digit_height * 4) / 5:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box = image[y:y + h, x:x + w]
            name_box = file_name + '_' + str(i) + '_ctn.png'
            cv2.imwrite(name_box, box)
            # toA4(name_box)
            file_name_pad = AddPadding(name_box)
            image_pad = cv2.imread(file_name_pad)
            digit = getDigitFromImageSimple(image_pad)

            details.append(digit)


def AutCanny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def RotateTransparent(res_file, angle=0):
    img = Image.open(res_file)
    # converted to have an alpha layer
    im2 = img.convert('RGBA')
    # rotated image
    rot = im2.rotate(angle, expand=1)
    # a white image same size as rotated image
    fff = Image.new('RGBA', rot.size, (255,) * 4)
    # create a composite image using the alpha layer of rot as a mask
    out = Image.composite(rot, fff, rot)
    # save your work (converting back to mode='1' or whatever..)
    # out.convert(img.mode).save(res_file_true)
    return out


def toA4(image_file):
    from PIL import Image

    im = Image.open(image_file)
    a4im = Image.new('RGB',
                     (595, 842),  # A4 at 72dpi
                     (255, 255, 255))  # White
    a4im.paste(im, im.getbbox())  # Not centered, top-left corner
    a4im.save(image_file + 'a4.pdf', 'PDF', quality=100)


def AddPadding(img_file):
    old_im = Image.open(img_file)
    old_size = old_im.size

    new_size = (50, 50)
    new_im = Image.new("RGB", new_size, (255, 255, 255))
    new_im.paste(old_im, ((new_size[0] - old_size[0]) / 2,
                          (new_size[1] - old_size[1]) / 2))

    res_file = img_file + '_padding.png'
    new_im.save(res_file)
    return res_file


def writeText(data):
    text_file = open("result.txt", "w")
    text_file.write(data)
    text_file.close()
