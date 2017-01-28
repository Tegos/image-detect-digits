import cv2


def get_line_param(p1, p2):
    x1 = float(p1[0])
    y1 = float(p1[1])

    x2 = float(p2[0])
    y2 = float(p2[1])

    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    return k, b


def draw_full_line(point1, point2, img):
    # plot the secant
    k, b = get_line_param(point1, point2)
    print point1, point2
    print k, b
    height, width, ch = img.shape

    x1 = 0
    y1 = k * x1 + b

    x2 = width
    y2 = k * x2 + b

    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))

    print p1, p2

    # c = intercept
    # x_min, x_max = axes.get_xlim()
    # y_min, y_max = c, c + slope * (x_max - x_min)
    #
    # data_y = (int(x[0] * slope + intercept), int(x[1] * slope + intercept))
    # x = list(x)
    # x = np.array(x)
    # x = x.astype(int)
    # x = tuple(x)
    #

    # # cv2.line(img, (int(x_min), int(x_max)), (int(y_min), int(y_max)), (0, 255, 0), 3)
    cv2.line(img, p1, p2, (0, 255, 255), 2)
