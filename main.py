import numpy as np
from PIL import Image
import cv2
from typing import Literal
from harris import harris
from susan import susan
from sift import sift
from datetime import datetime
#delete later
#import ftdetect.features
import matplotlib.pyplot as plt
#######################
def timer(im, function, *args):
  start = datetime.now()
  function(im, *args)
  end = datetime.now()
  return end - start

def test_stability(img1, img2, susan_hypers = [3, 27, 14.5], harris_hypers = [0.07, 5]):
  sus1 = susan(img1, *susan_hypers)
  sus2 = susan(img2, *susan_hypers)
  har1 = harris(img1, *harris_hypers)
  har2 = harris(img2, *harris_hypers)
  print("stab_factor is",calculate_stability(sus1, sus2))
  # calculate stability factor of image_4 and image_5 on harris
  h1 = np.zeros(har1.shape)
  h1[har1 > 0.01 * har1.max()] = 1
  h2 = np.zeros(har2.shape)
  h2[har2 > 0.01 * har2.max()] = 1
  print("stab_factor is", calculate_stability(h1, h2))

def calculate_stability(arr1, arr2):
  intersect= np.count_nonzero(np.logical_and(arr1,arr2))
  elem1 = np.count_nonzero(arr1)
  elem2 = np.count_nonzero(arr2)
  min_elem= min(elem1, elem2)
  return (intersect / min_elem) * 100

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

  #noise_dino1 = Image.open('images/noise_dino_1.png').convert('L')
  #noise_dino_arr1 = np.asarray(noise_dino1)
  print(timer(img_arr4, susan, 3, 27, 14.5))
  
  #filter_image(sus4, img_arr4, type='sus')
  #sus5 = susan(img_arr5, 3, 27, 14.5)
  #filter_image(sus5, img_arr5, type='sus')

  #sus1 = susan(dino_arr1, 3, 27, 14.5)
  #filter_image(sus1, dino_arr1, type='sus')
  #sus2 = susan(dino_arr2, 3, 27, 14.5)
  #filter_image(sus2, dino_arr2, type='sus')

  #sus_noise1 = susan(noise_dino_arr1, 3, 40, 14.5)
  #filter_image(sus_noise1, noise_dino_arr1, type='sus')



  h4= harris(img_arr4, 0.07, 5)
  #filter_image(h4, img_arr4)
  h5 = harris(img_arr5, 0.07, 5)
  #filter_image(h5, img_arr5)

  #h1= harris(dino_arr1, 0.07, 5)
  #filter_image(h1, dino_arr1)
  #h2 = harris(dino_arr2, 0.07, 5)
  #filter_image(h2, dino_arr2)

  #h_noise_1 = harris(noise_dino_arr1, 0.1, 5)
  #filter_image(h_noise_1, noise_dino_arr1)

  #calculate stability factor of image_4 and image_5 on susan
  test_stability(img_arr4, img_arr5)
  #sif = sift(img_arr)
