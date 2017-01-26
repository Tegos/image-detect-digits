import numpy as np
import cv2

source = cv2.imread('../res/555.png')
object = cv2.imread('../res/search.png')
result = cv2.matchTemplate(source, object, cv2.TM_CCOEFF_NORMED)
result_ = cv2.matchTemplate(source, object, 1)
(y, x) = np.unravel_index(result.argmax(), result.shape)
(y1, x1) = np.unravel_index(result_.argmax(), result_.shape)
print x, y
print x1, y1
