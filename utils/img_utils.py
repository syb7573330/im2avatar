import numpy as np
import scipy as sp
from PIL import Image
from scipy import misc

def imresize(np_image, new_dims):
  """
  Args:
    np_image: numpy array of dimension [height, width, 3], intensity ranges from 0-255
    new_dims: A python list containing the [height, width], number of rows, columns.
  Returns:
    im: numpy array resized to dimensions specified in new_dims.
  """
  im = np.uint8(np_image)
  im = Image.fromarray(im) 
  new_height, new_width = new_dims
  im = im.resize((new_width, new_height), Image.ANTIALIAS)
  return np.array(im)

def imread(filename, new_dims=None):
  """ Read image and add an extra dimension outside
      new_dims: A python list containing the [height, width], number of rows, columns.
  """
  im = sp.misc.imread(filename) # type 'numpy.ndarray'

  if new_dims is None:
    return im / 255.0
  else:
    return imresize(im, new_dims) / 255.0

def imsave(np_image, filename):
  """Save image to file.
  """
  im = sp.misc.toimage(np_image, cmin=0, cmax=1.0)
  im.save(filename)
