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
  img4 = Image.open('images/sens_1.jpg').convert('L')
  img5 = Image.open('images/sens_2.jpg').convert('L')
  # make the image numpy array
  img_arr4 = np.asarray(img4)
  img_arr5 = np.asarray(img5)

  #dino1 = Image.open('images/dino_1.png').convert('L')
  #dino_arr1 = np.asarray(dino1)
  #dino2 = Image.open('images/dino_2.png').convert('L')
  #dino_arr2 = np.asarray(dino2)

  noise_dino1 = Image.open('images/noise_sens_1.png').convert('L')
  noise_dino_arr1 = np.asarray(noise_dino1)

  sus4 = susan(img_arr4, 3, 27, 14.5)
  filter_image(sus4, img_arr4, type='sus')
  #sus5 = susan(img_arr5, 3, 27, 14.5)
  #filter_image(sus5, img_arr5, type='sus')

  #sus1 = susan(dino_arr1, 3, 27, 14.5)
  #filter_image(sus1, dino_arr1, type='sus')
  #sus2 = susan(dino_arr2, 3, 27, 14.5)
  #filter_image(sus2, dino_arr2, type='sus')

  sus_noise1 = susan(noise_dino_arr1, 3, 40, 14.5)
  filter_image(sus_noise1, noise_dino_arr1, type='sus')

  #h4= harris(img_arr4, 0.07, 5)
  #filter_image(h4, img_arr4)
  #h5 = harris(img_arr5, 0.07, 5)
  #filter_image(h5, img_arr5)

  #h1= harris(dino_arr1, 0.07, 5)
  #filter_image(h1, dino_arr1)
  #h2 = harris(dino_arr2, 0.07, 5)
  #filter_image(h2, dino_arr2)

  #h_noise_1 = harris(noise_dino_arr1, 0.1, 5)
  #filter_image(h_noise_1, noise_dino_arr1)
  """
  #calculate stability factor of image_4 and image_5 on susan
  intersect= np.count_nonzero(np.logical_and(sus4,sus5))
  elem_sus4 = np.count_nonzero(sus4)
  elem_sus5 = np.count_nonzero(sus5)
  min_elem= min(elem_sus4,elem_sus5)
  stab_fact = (intersect / min_elem) * 100
  print("stab_factor is",stab_fact)

  # calculate stability factor of image_4 and image_5 on harris
  h4_arr = np.zeros(h4.shape)
  h4_arr[h4 > 0.01 * h4.max()] = 1
  h5_arr = np.zeros(h5.shape)
  h5_arr[h5 > 0.01 * h5.max()] = 1
  intersect2 = np.count_nonzero(np.logical_and(h4_arr,h5_arr))
  elem_h4 = np.count_nonzero(h4_arr)
  elem_h5 = np.count_nonzero(h5_arr)
  min_elemh = min(elem_h4, elem_h5)
  stab_facth = (intersect2/ min_elemh) * 100
  print("stab_factor is", stab_facth)
  """
  #calculate noise factor of sense1 for susan
  intersect = np.count_nonzero(np.logical_and(sus4, sus_noise1))
  elem_sus4 = np.count_nonzero(sus4)
  elem_sus5 = np.count_nonzero(sus_noise1)
  min_elem = min(elem_sus4, elem_sus5)
  noise_fact = (intersect / min_elem) * 100
  print("noise_factor is", noise_fact)




  #sif = sift(img_arr)
