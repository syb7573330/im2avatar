import numpy as np
import h5py

def load_h5_vol(path):
  """ Load 3D color volume from h5
  """
  f = h5py.File(path)
  return f['data'][:] 

def upsample_vol(vol, factor=2):
  """ vol: (depth, height, width, dim)
      upsample to (depth * factor, height * factor, width * factor, dim) using nearest neighbor upsampling scheme.
  """
  return vol.repeat(factor, axis=0).repeat(factor, axis=1).repeat(factor, axis=2)
