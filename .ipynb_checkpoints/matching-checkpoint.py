from operator import imod
import numpy as np
import cv2
from hu_moments import hu

# yalnizca gercek erkeklerin belli oldugu yer
def getInvariantMoments(img):
    moments = cv2.moments(img)
    huMoments = cv2.HuMoments(moments)
    for i in range(0,7):
        huMoments[i] = -1* np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    return huMoments

def match(img1, img2):
  inv1 = getInvariantMoments(img1)
  inv2 = getInvariantMoments(img2)
  print(inv1)
  print(inv2)

def imp_match(img1, img2):
  hu1 = hu(img1)
  hu2 = hu(img2)

  for i in range(0,7):
    hu1[i] = (-1* np.copysign(1.0, hu1[i]) * np.log10(abs(hu1[i]))).reshape((1,))
    hu2[i] = (-1* np.copysign(1.0, hu2[i]) * np.log10(abs(hu2[i]))).reshape((1,))
  print(hu1)
  print(hu2)