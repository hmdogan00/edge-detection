import numpy as np
from PIL import Image
import cv2
from typing import Literal
from harris import harris
from susan import susan
from sift import sift
#delete later
import ftdetect.features
import matplotlib.pyplot as plt
#######################

def filter_image(arr: np.ndarray, org: np.ndarray=[], title: str='Image', type: Literal['har', 'sus', 'sif'] = 'har'):
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
  if type == 'har':
    im[arr > 0.01 * arr.max()] = [255, 0, 0]
  elif type == 'sus':
    im[arr != 0] = [255, 0, 0]
  elif type == 'sif':
    im[arr != 0] = [255, 0, 0]
  im = Image.fromarray(im)
  im.show(title=title)

if __name__ == '__main__':
  # get image using PIL and convert to grayscale
  img1 = Image.open('images/image_1.jpg').convert('L')
  img2 = Image.open('images/image_2.png').convert('L')
  img3 = Image.open('images/image_3.jpg').convert('L')
  # make the image numpy array
  img_arr1 = np.asarray(img1)
  img_arr2 = np.asarray(img2)
  img_arr3 = np.asarray(img3)
  """""
  sus1 = susan(img_arr2, 3, 27, 14.5)
  filter_image(sus1, img_arr2, type='sus')
  sus2 = susan(img_arr2, 3, 27, 14.5)
  filter_image(sus2, img_arr2, type='sus')
  sus3 = susan(img_arr3, 3, 27, 14.5)
  filter_image(sus3, img_arr3, type='sus')
  
  # library

  xoffset, yoffset = 0.975, 0.08
  xpos, ypos = xoffset * img_arr4.shape[1], yoffset * img_arr4.shape[0]
  sucorner = ftdetect.features.susanCorner(img_arr4)
  # Plot the results of the SUSAN corner detector
  fig = plt.figure()
  fig.canvas.set_window_title('images/image_4.jpg')
  ax = fig.add_axes((0.51, 0.01, 0.48, 0.48))
  ax.set_axis_off()
  ax.imshow(img4, interpolation='nearest', cmap='Greys_r')
  ax.autoscale(tight=True)
  vidx, hidx = sucorner.nonzero()
  ax.plot(hidx, vidx, 'bo')
  plt.text(xpos, ypos, 'SUSAN corners', color='r')
  plt.show()
  """""
  h1 = harris(img_arr1, 0.07, 5)
  filter_image(h1, img_arr1)
  h2 = harris(img_arr2, 0.07, 5)
  filter_image(h2, img_arr2)
  h3 = harris(img_arr3, 0.07, 5)
  filter_image(h3, img_arr3)



  #sif = sift(img_arr)
