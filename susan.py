import numpy as np
import math


def create_circular_mask(r: int):
    """Creates a circular mask with given radius.

  Parameters
  ----------
  r : int
    Radius of the mask

  Returns
  -------
  mask : ndarray
    Circular mask with given radius

  Examples
  --------
  r = 3 returns:
  [[0. 0. 0. 1. 0. 0. 0.]
   [0. 1. 1. 1. 1. 1. 0.]
   [0. 1. 1. 1. 1. 1. 0.]
   [1. 1. 1. 1. 1. 1. 1.]
   [0. 1. 1. 1. 1. 1. 0.]
   [0. 1. 1. 1. 1. 1. 0.]
   [0. 0. 0. 1. 0. 0. 0.]]
  """
    mask = np.zeros((r * 2 + 1, r * 2 + 1))
    x, y = np.ogrid[-r:r + 1, -r: r + 1]
    mask[x ** 2 + y ** 2 <= r ** 2] = 1
    return mask


def susan(img: np.ndarray, mask_radius: int, t: int, g: int):
    """Calculates SUSAN Corner Detection array for every element (pixel) in the given ndarray (image)

  SUSAN Corner Detector: R = g - n(r_0) if n(r) < g, 0 if n(r) >= g

  where n(r_0) = sum( c(r, r_0) )
  where c(r, r_0) = exp( - ( (Ir - Ir_0) / t )^6 )
  where Ir is the gray value of a pixel,
  t is the gray difference threshold which determines the anti-noise ability,
  g is the geometric threshold which determines the acute level of a corner

  Parameters
  ----------
  img : ndarray
    Image to get its corners calculated

  mask_radius : int
    Radius of the circular mask that will be used to calculate the n(r_0) value of the pixel.

  t : int
    Gray difference threshold value: increases anti-noise.

  g : int
    Geometric threshold value: enables more acute corners as g gets smaller.

  Returns
  -------
  output : ndarray
    R values of the pixels
  """
    # initialize mask and output array
    mask = create_circular_mask(mask_radius)
    output = np.zeros(img.shape)
    # calculate the max range of the pixels
    # iterate through each pixel of the image
    for row_ind in range(len(img)):
        for col_ind in range(len(img[0])):
            # find the column values that can be intersection of mask and image
            # find the min column value for the intersection of (row_ind,col_ind) pixel of the image and the mask
            temp_col = col_ind - math.floor(
                len(mask[0]) / 2)  # divide by two since nucleus will be iterated over pixels
            min_col = max(temp_col, 0)
            # find the max column value for the intersection of (row_ind,col_ind) pixel of the image and the mask
            temp_col2 = col_ind + math.floor(
                len(mask[0]) / 2)  # divide by two since nucleus will be iterated over pixels
            max_col = temp_col2
            # if the number of max column exceeds the image boundary, limit it
            if max_col >= len(img[0]):
                max_col = len(img[0]) - 1
            # find the row values that can be intersection of structure element and the image
            # find the min row value for the intersection of (row_ind,col_ind) pixel of the image and the mask
            temp_row = row_ind - math.floor(
                len(mask) / 2)  # divide by two since nucleus will be iterated over pixels
            min_row = max(temp_row, 0)
            # find the max row value for the intersection of (row_ind,col_ind) pixel of the image and the mask
            temp_row2 = row_ind + math.floor(
                len(mask) / 2)   # divide by two since nucleus will be iterated over pixels
            max_row = temp_row2
            # if the number of max row exceeds the image boundary, limit it
            if max_row >= len(img):
                max_row = len(img) - 1
            # find I(r0) value = nucleus' pixel value on image
            i_r0 = img[row_ind, col_ind]
            # initialize USAN area to 0
            n_r0 = 0
            # iterate over mask
            for y in range(min_row, max_row+1):
                for x in range(min_col, max_col+1):
                    # find the other pixel values within the mask
                    i_r = img[y][x]
                    # calculate the comparision formula
                    c_r_r0 = math.exp(-((float(i_r) - float(i_r0)) / t) ** 6)
                    # sum up to the USAN area
                    n_r0 = n_r0 + c_r_r0
            # Whether the USAN area is smaller than the geometric threshold, or not
            if n_r0 < g:
                r_r0 = g - n_r0
            else:
                r_r0 = 0
            # put it to the response to corner to the output matrix
            output[row_ind, col_ind] = r_r0
    return output
