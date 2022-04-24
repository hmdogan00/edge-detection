from harris import harris
from susan import susan
import numpy as np
import cv2
from hu_moments import hu
import pandas as pd
import matplotlib.pyplot as plt

def get_one_image(img_list):
    max_width = 0
    total_height = 200  # padding
    for img in img_list:
        if img.shape[1] > max_width:
            max_width = img.shape[1]
        total_height += img.shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height, max_width), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        image = np.hstack((image, np.zeros((image.shape[0], max_width - image.shape[1]))))
        final_image[current_y:current_y + image.shape[0], :] = image
        current_y += image.shape[0]
    return final_image

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

def get_feature_hus(img, params=[0.07, 5]):
  #har = harris(img, *params)
  #h1 = np.where(har > 0.01 * har.max())
  har = susan(img, 3, 27, 14.5)
  h1 = np.where(har != 0)

  rows = []
  cols = []
  hus = []
  for i in range(len(h1[0])):
      row = h1[0][i]
      col = h1[1][i]
      if row - 3 < 0 or row + 4 >= img.shape[0] or col - 3 < 0 or col + 4 >= img.shape[1]:
          continue
      sub_img = img[row - 3: row + 4,col - 3: col + 4]
      hu1 = getInvariantMoments(sub_img)
      
      rows.append(row)
      cols.append(col)
      hus.append(hu1)
  data = {
      'rows': rows,
      'cols': cols,
      'hus': hus
  }
  return pd.DataFrame(data)

def imp_match(img1, img2):
  print('getting huus')
  hu1 = get_feature_hus(img1)
  hu2 = get_feature_hus(img2)
  
  print('getting distances...')
  indices1 = []
  indices2 = []
  hu1s = []
  hu2s = []
  ratios = np.zeros(hu1['hus'].shape)

  for i, hus1 in enumerate(hu1['hus']):
      distances = np.zeros(hu2['hus'].shape)
      for k, hus2 in enumerate(hu2['hus']):
          distances[k] = np.linalg.norm(hus2 - hus1)
          
      min_index = np.argmin(distances)
      min_val = np.min(distances)
      distances[min_index] = np.inf
      second_min_index = np.argmin(distances)
      second_min_val = np.min(distances)
      
      ratios[i] = min_val / second_min_val
      
      indices1.append([hu1['rows'][i], hu1['cols'][i]])
      indices2.append([hu2['rows'][min_index], hu2['cols'][min_index]])
      hu1s.append(hus1)
      hu2s.append(min_val)
  df = pd.DataFrame({
      'index1': indices1,
      'index2': indices2,
      'hu1s': hu1s,
      'hu2s': hu2s,
      'ratios': ratios
  })
  print(df)
  print('mean and output image')
  mean = np.mean(df['ratios'][df['ratios'] > -np.inf])
  if np.isnan(mean):
      mean = 0
  goods = df[df['ratios'] < (mean) ]
  first_ind = goods['index1']
  second_ind = goods['index2']

  output = get_one_image([img1,img2])
  offset = img1.shape[0]
  print('drawing lines')
  img = output[:,:]
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  for i in enumerate(first_ind):
      try:
          r = np.random.randint(255)
          g = np.random.randint(255)
          b = np.random.randint(255)
          img = cv2.line(img, (first_ind[i][1], first_ind[i][0]), (second_ind[i][1], second_ind[i][0] + offset), (r,g,b), 1)
      except KeyError:
          continue
  print('geldik sonunda')
  plt.imshow(img)