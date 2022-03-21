from matplotlib.pyplot import title
import numpy as np
from PIL import Image
import cv2
from harris import harris
from susan import susan

def filter_image(arr:np.ndarray, org:np.ndarray=[], title:str='Image'):
  """Filters given image array (has to be numpy array) and prints it

  Parameters
  ----------
  arr : ndarray 
    Filter array
  org : ndarray 
    Original array to be filtered - defaults to black image
  title : str 
    Title of the printed image
  """
  if len(org) == 0:
    org = np.zeros(arr.shape)
  im = org[:,:]
  im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
  im[arr > 0.01 * arr.max()] = [255, 0, 0]
  im = Image.fromarray(im)
  im.show(title=title)

if __name__ == '__main__':
  # get image using PIL and convert to grayscale
  img = Image.open('images/image_1.jpg').convert('L')
  # make the image numpy array
  img_arr = np.asarray(img)

  # TODO: implement SUSAN algorithm
  h = harris(img_arr, 0.07, 2)
  filter_image(h, img_arr)
  s = susan(img_arr)