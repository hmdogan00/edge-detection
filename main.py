import numpy as np
from PIL import Image
import cv2
from typing import Literal
from datetime import datetime

from harris import harris
from susan import susan
from matching import match, imp_match

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
  print("Stability factor of SUSAN is",calculate_stability(sus1, sus2))
  # calculate stability factor of image_4 and image_5 on harris
  h1 = np.zeros(har1.shape)
  h1[har1 > 0.01 * har1.max()] = 1
  h2 = np.zeros(har2.shape)
  h2[har2 > 0.01 * har2.max()] = 1
  print("Stability factor of Harris is", calculate_stability(h1, h2))

def calculate_stability(arr1, arr2):
  intersect= np.count_nonzero(np.logical_and(arr1,arr2))
  elem1 = np.count_nonzero(arr1)
  elem2 = np.count_nonzero(arr2)
  min_elem= min(elem1, elem2)
  return (intersect / min_elem) * 100

def test_noise(img, noised):
  sus = susan(img, 3, 40, 14.5)
  noise_sus = susan(noised, 3, 40, 14.5)
  har = harris(img, 0.07, 5)
  noise_har = harris(noised, 0.07, 5)
  print('Noise factor of SUSAN is', calculate_noise(sus, noise_sus))
  print('Noise factor of Harris is', calculate_noise(har, noise_har))

def calculate_noise(img, noised):
  intersect = np.count_nonzero(np.logical_and(img, noised))
  elem1 = np.count_nonzero(img)
  elem2 = np.count_nonzero(noised)
  min_elem = min(elem1, elem2)
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
  img1 = cv2.imread('images/plane.jpeg',cv2.IMREAD_GRAYSCALE)
  img2 = cv2.imread('images/plane_90deg.jpeg',cv2.IMREAD_GRAYSCALE)

  imp_match(img1, img2)
  
  #print(timer(img_arr4, susan, 3, 27, 14.5))
  #print(timer(img_arr4, harris, 0.07, 5))
  
  #noise_dino1 = Image.open('images/noise_sens_1.png').convert('L')
  #noise_dino_arr1 = np.asarray(noise_dino1)

  #sus4 = susan(img_arr4, 3, 27, 14.5)
  #filter_image(sus4, img_arr4, type='sus')

  #sus_noise1 = susan(noise_dino_arr1, 3, 40, 14.5)
  #filter_image(sus_noise1, noise_dino_arr1, type='sus')

  #calculate stability factor of image_4 and image_5 on susan
  #test_stability(img_arr4, img_arr5)

  #calculate noise factor of sense1 for susan
  #test_noise(img_arr4, noise_dino_arr1)

  #sif = sift(img_arr)
  # calculate noise factor of sense1 for susan
"""   intersect = np.count_nonzero(np.logical_and(sus4, sus_noise1))
  elem_sus4 = np.count_nonzero(sus4)
  elem_sus5 = np.count_nonzero(sus_noise1)
  min_elem = min(elem_sus4, elem_sus5)
  noise_fact = (intersect / min_elem) * 100
  print("noise_factor is", noise_fact) """
