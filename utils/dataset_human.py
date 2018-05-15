import numpy as np
import os
import glob
from random import shuffle
import img_utils
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/')
if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)

if not os.path.exists(os.path.join(DATA_DIR, 'human_im2avatar')):
  www = 'https://www.dropbox.com/s/imgiu8xump2zlvm/human_im2avatar.tar.gz'
  zipfile = os.path.basename(www)
  os.system('wget %s; tar -xzf %s' % (www, zipfile))
  os.system('mv %s %s' % (zipfile[:zipfile.find('.')], DATA_DIR))
  os.system('rm %s' % (zipfile))

class Dataset(object):
  def __init__(self, base_path, data_list_path):
    """ base_path: the path to data (colorful 3D volume, views and flow volume)
        data_list_path: the path to the folder containing train_ or test_list_human.txt
    """
    self.base_path = base_path
    self.data_list_path = data_list_path

    self.train_names = self.loadNames("train") # [[cloth, mesh], ...]
    self.test_names = self.loadNames("test")

  def loadNames(self, process):
    res = []
    path_ = os.path.join(self.data_list_path, "{}_list_human.txt".format(process))
    with open(path_, 'r') as file_:
      res = file_.readlines()
    res = [tmp.rstrip().split() for tmp in res]
    return res

  def shuffleTrainNames(self): 
    shuffle(self.train_names)

  def getTrainSampleSize(self):
    return len(self.train_names)

  def getTestSampleSize(self):
    return len(self.test_names)

  def getName(self, process, idx):
    """return [cloth, mesh]"""
    res = ''
    if process == 'train':
      res = self.train_names[idx]
    elif process == 'test':
      res = self.test_names[idx]
    return res

  def next_batch(self, start_idx, batch_size, shuffle_view=True):
    """ Get the next training batch.
    """
    tmp_names = self.train_names[start_idx:start_idx+batch_size]

    res_imgs = [] # [batch_size, (im_height, im_width, 3)]
    res_vols = [] # [batch, (vol_depth, vol_height, vol_width, 3)]
  
    for name in tmp_names:
      cloth = name[0]
      mesh = name[1]

      # Load image
      view_path = os.path.join(self.base_path, cloth, mesh, "views/*.png")
      im_paths = glob.glob(view_path)
      if shuffle_view:
        shuffle(im_paths)
      im_path = im_paths[0]
      im = img_utils.imread(im_path)
      res_imgs.append(im)

      # Load volume
      vol_path = os.path.join(self.base_path, cloth, mesh, "{}.h5".format(mesh))
      f = h5py.File(vol_path)
      vol = f['data'][:]
      res_vols.append(vol)

    return np.array(res_imgs), np.array(res_vols)

  def next_flow_batch(self, start_idx, batch_size, vol_dim=64, shuffle_view=True):
    """ Get the next training batch with flow data.
    """
    tmp_names = self.train_names[start_idx:start_idx+batch_size]

    res_imgs = [] # [batch_size, (im_height, im_width, 3)]
    res_flow_vols = [] # [batch, (vol_depth, vol_height, vol_width, 2)]
    res_clr_vols = [] # [batch, (vol_depth, vol_height, vol_width, 3)]

    for name in tmp_names:
      cloth = name[0]
      mesh = name[1]

      # Load image
      view_path = os.path.join(self.base_path, cloth, mesh, "views/*.png")
      im_paths = glob.glob(view_path)
      if shuffle_view:
        shuffle(im_paths)
      im_path = im_paths[0]
      im = img_utils.imread(im_path)
      res_imgs.append(im)

      #####################
      # Load flow volume
      #####################
      im_name = os.path.basename(im_path)[:-4]
      flow_path = os.path.join(self.base_path, cloth, mesh, "flow/{}_{}_coor.h5".format(vol_dim, im_name))
      f = h5py.File(flow_path)
      flow_vol = f['data'][:]
      res_flow_vols.append(flow_vol)

      ####################
      # Load clor volume
      ####################
      vol_path = os.path.join(self.base_path, cloth, mesh, "{}.h5".format(mesh))
      f = h5py.File(vol_path)
      vol = f['data'][:]
      res_clr_vols.append(vol)

    return np.array(res_imgs), np.array(res_flow_vols), np.array(res_clr_vols) 


  def next_test_batch(self, start_idx, batch_size):
    tmp_names = self.test_names[start_idx:start_idx+batch_size]

    res_imgs = [] # [view_size, (im_height, im_width, 3)]
    res_img_names = [] # [[cloth, mesh, view], ...]
    
    for name in tmp_names:
      cloth = name[0]
      mesh = name[1]

      view_path = os.path.join(self.base_path, cloth, mesh, "views/*.png")
      im_paths = glob.glob(view_path)
      
      for im_path in im_paths:
        im = img_utils.imread(im_path)
        res_imgs.append(im)

        res_img_names.append([cloth, mesh, os.path.basename(im_path)])

    return np.array(res_imgs), res_img_names








