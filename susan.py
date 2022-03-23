import numpy as np

def create_circular_mask( r: int ):
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
  x, y = np.ogrid[-r:r+1, -r: r+1]
  mask[x**2+y**2 <= r**2] = 1
  return mask
  
def susan( img: np.ndarray, mask_radius: int, t: int, g: int ):
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
  #initialize mask and output array
  mask = create_circular_mask(mask_radius)
  output = np.zeros(img.shape)

  # calculate the max range of the pixels
  x_range, y_range = img.shape
  y_range -= mask_radius
  x_range -= mask_radius

  for x in range(mask_radius, x_range):
    for y in range(mask_radius, y_range):
      i_r = img[ (x-mask_radius):(x+mask_radius+1), (y-mask_radius):(y+mask_radius+1) ]
      i_r = i_r[mask == 1]
      i_r0 = img[x, y]
      c_r_r0 = np.sum(np.exp(-((i_r - i_r0) / t) ** 6))
      if c_r_r0 < g:
        c_r_r0 = g - c_r_r0
      else:
        c_r_r0 = 0
      output[x, y] = c_r_r0
  return output