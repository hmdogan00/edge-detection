import numpy as np

def harris(img : np.ndarray, k : int, window_size : int):
  """Calculates Harris Corner Detection array for every element (pixel) in the given ndarray (image)

  Harris Detector: R = det(H) - k(trace(H))**2

  where trace(H) = eigval_1 + eigval_2
  and det(H) = eigval_1 * eigval_2

  the eigenvalues are of H's eigenvalues.

  Parameters
  ----------
  img : ndarray
    Image to get its corners calculated
  
  k : int
    k value to weight the trace
  
  Returns
  -------
  output : ndarray
    R values of the pixels
  """
  # initialize output array and offset
  output = np.zeros(img.shape)
  offset = int(window_size / 2)

  # calculate the max range of the pixels
  y_range, x_range = img.shape
  y_range -= offset
  x_range -= offset

  # calculate the gradient of all pixels
  dy, dx = np.gradient(img)
  Ixx = dx * dx
  Ixy = dx * dy
  Iyy = dy * dy

  for y in range(offset, y_range):
    for x in range(offset, x_range):

      # limits of the window
      y_start = y - offset
      x_start = x - offset
      y_end = y + offset + 1
      x_end = x + offset + 1

      # calculate values of H, which is [[I_xx, I_xy], [I_yx, I_yy]]
      i_xx = Ixx[y_start : y_end, x_start : x_end]
      i_xy = Ixy[y_start : y_end, x_start : x_end]
      i_yy = Iyy[y_start : y_end, x_start : x_end]

      # summation to find the intensities
      s_xx = np.sum(i_xx)
      s_xy = np.sum(i_xy)
      s_yy = np.sum(i_yy)

      # compute determinant and trace of H
      det = (s_xx * s_yy) - (s_xy * s_xy)
      trace = s_xx + s_yy

      # put the R value of the pixel
      output[y,x] = det - k * trace**2

  return output