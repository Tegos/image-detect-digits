import subprocess

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
import os
import sys

img_path = 'C:\\test.png'
text = pytesseract.image_to_string(Image.open(img_path))
print img_path

mypath = os.path.abspath(__file__)
mydir = os.path.dirname('C:\\Python27\\')
start = os.path.join(mydir, "start.py")
subprocess.call([sys.executable, start])
