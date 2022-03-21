from matplotlib.pyplot import title
import numpy as np
from PIL import Image
import cv2
from harris import harris
from susan import susan

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
  # get image using PIL and convert to grayscale
  img = Image.open('images/image_1.jpg').convert('L')
  # make the image numpy array
  img_arr = np.asarray(img)
  
  # get correct harris filter and print it to check our results
  r_harris = real_harris(img_arr)
  print_image(r_harris, org=img_arr , title='Real Harris Detection')

  # TODO: implement harris and SUSAN algorithms
  h = harris(img_arr)
  s = susan(img_arr)