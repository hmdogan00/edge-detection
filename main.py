from matplotlib.pyplot import title
import numpy as np
from PIL import Image
import cv2
from harris import harris

def real_harris(img):
  operatedImage = np.float32(img)

  dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
  dest = cv2.dilate(dest, None)
  return dest

def print_image(arr, org=[], title='Image'):
  if len(org) == 0:
    org = np.zeros(arr.shape)
  im = org[:,:]
  im[arr > 0.01 * arr.max()] = [255]
  im = Image.fromarray(im)
  im.show(title=title)

if __name__ == '__main__':
  img = Image.open('images/image_1.jpg').convert('L')
  img_arr = np.asarray(img)
  
  r_harris = real_harris(img_arr)

  print_image(r_harris, org=img_arr , title='Real Harris Detection')

  h = harris(img_arr)
