import numpy as np
import os
import glob
from random import shuffle
import img_utils
import vol_utils
import h5py
from scipy.spatial.distance import cdist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/')
if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)

if not os.path.exists(os.path.join(DATA_DIR, 'ShapeNetCore_im2avatar')):
  www = 'https://www.dropbox.com/s/s03fc1rx4ljkhix/ShapeNetCore_im2avatar.tar.gz'
  zipfile = os.path.basename(www)
  os.system('wget %s; tar -xzf %s' % (www, zipfile))
  os.system('mv %s %s' % (zipfile[:zipfile.find('.')], DATA_DIR))
  os.system('rm %s' % (zipfile))


class Dataset(object):
  def __init__(self, base_path, cat_id, data_list_path):
    """base_path: The path to the category 
       cat_id: i.e. '02958343', 
       data_list_path: The path to the folder containing 'train_list_{cat_id}.txt/val_list_{cat_id}.txt/test_list_{cat_id}.txt'
    """
    self.base_path = base_path
    self.cat_id = cat_id
    self.data_list_path = data_list_path

    # find all the individual ids
    self.trainIds = self.loadIds("train")
    self.valIds = self.loadIds("val")
    self.testIds = self.loadIds("test")

  def loadIds(self, process):
    """ process: "train", "val" or "test"
    """
    res = []
    path_ = os.path.join(self.data_list_path, "{}_list_{}.txt".format(process, self.cat_id))
    with open(path_, 'r') as file_:
      res = file_.readlines()
    res = [tmp.rstrip() for tmp in res]
    return res

  def shuffleIds(self): # Shuffle training ids 
    shuffle(self.trainIds)

  def getTrainSampleSize(self):
    return len(self.trainIds)

  def getValSampleSize(self):
    return len(self.valIds)

  def getTestSampleSize(self):
    return len(self.testIds)

  def getId(self, process, idx):
    res = ''
    if process == 'train':
      res = self.trainIds[idx]
    elif process == 'val':
      res = self.valIds[idx]
    elif process == 'test':
      res = self.testIds[idx]
    return res

  def next_batch(self, start_idx, batch_size, vol_dim=64, process="train", shuffle_view=True):
    tmp_ids = []
    if process == "train":
      tmp_ids = self.trainIds[start_idx:start_idx+batch_size]
    elif process == "val":
      tmp_ids = self.valIds[start_idx:start_idx+batch_size]
    elif process == "test":
      tmp_ids = self.testIds[start_idx:start_idx+batch_size]


    res_imgs = [] # [batch_size, (im_height, im_width, 3)]
    res_vols = [] # [batch, (vol_depth, vol_height, vol_width, 3)]
  
    for ins_id in tmp_ids:
      # Load image
      view_path = os.path.join(self.base_path, self.cat_id, ins_id, "views/*.png")
      im_paths = glob.glob(view_path)
      if shuffle_view:
        shuffle(im_paths)
      im_path = im_paths[0]
      im = img_utils.imread(im_path)
      res_imgs.append(im)

      # Load volume
      vol_path = os.path.join(self.base_path, self.cat_id, ins_id, "models/model_normalized_{}.h5".format(vol_dim))
      f = h5py.File(vol_path)
      vol = f['data'][:]
      res_vols.append(vol)

    return np.array(res_imgs), np.array(res_vols)

  
  def next_flow_batch(self, start_idx, batch_size, vol_dim=64, process='train', shuffle_view=True):
    tmp_ids = []
    if process == "train":
      tmp_ids = self.trainIds[start_idx:start_idx+batch_size]
    elif process == "val":
      tmp_ids = self.valIds[start_idx:start_idx+batch_size]

    res_imgs = [] # [batch_size, (im_height, im_width, 3)]
    res_flow_vols = [] # [batch, (vol_depth, vol_height, vol_width, 2)]
    res_clr_vols = [] # [batch, (vol_depth, vol_height, vol_width, 3)]

    for ins_id in tmp_ids:
      ######################
      # Load image
      ######################
      view_path = os.path.join(self.base_path, self.cat_id, ins_id, "views/*.png")
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
      flow_path = os.path.join(self.base_path, self.cat_id, ins_id, "models/{}_{}_coor.h5".format(vol_dim, im_name))
      f = h5py.File(flow_path)
      flow_vol = f['data'][:]
      res_flow_vols.append(flow_vol)

      ####################
      # Load clor volume
      ####################
      vol_path = os.path.join(self.base_path, self.cat_id, ins_id, "models/model_normalized_{}.h5".format(vol_dim))
      f = h5py.File(vol_path)
      clr_vol = f['data'][:] # (vol_dim, vol_dim, vol_dim, 3)
      res_clr_vols.append(clr_vol)

    return np.array(res_imgs), np.array(res_flow_vols), np.array(res_clr_vols) 



  def next_test_batch(self, start_idx, batch_size):
    tmp_ids = self.testIds[start_idx:start_idx+batch_size]

    res_imgs = [] # [view_size, (im_height, im_width, 3)]
    res_img_names = [] # []
    
    for ins_id in tmp_ids:
      # Load image
      view_path = os.path.join(self.base_path, self.cat_id, ins_id, "views/*.png")
      im_paths = glob.glob(view_path)

      for im_path in im_paths:
        im = img_utils.imread(im_path)
        res_imgs.append(im)
        res_img_names.append(os.path.basename(im_path))

    return np.array(res_imgs), res_img_names
