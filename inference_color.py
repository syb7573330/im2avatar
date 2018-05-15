import tensorflow as tf
import numpy as np
import os
import h5py
import sys
sys.path.append('./utils')
sys.path.append('./models')

import dataset as dataset
import model_color as model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_color',
                           """Directory where to write summaries and checkpoint.""")
tf.app.flags.DEFINE_string('base_dir', './data/ShapeNetCore_im2avatar', 
                           """The path containing all the samples.""")
tf.app.flags.DEFINE_string('cat_id', '02958343', 
                           """The category id for each category: 02958343, 03001627, 03467517, 04379243""")
tf.app.flags.DEFINE_string('data_list_path', './data_list', 
                          """The path containing data lists.""")
tf.app.flags.DEFINE_string('output_dir', './output_color',
                           """Directory to save generated volume.""")

TRAIN_DIR = os.path.join(FLAGS.train_dir, FLAGS.cat_id)
OUTPUT_DIR = os.path.join(FLAGS.output_dir, FLAGS.cat_id)

if not os.path.exists(OUTPUT_DIR): 
  os.makedirs(OUTPUT_DIR)

# The views' size
BATCH_SIZE = 12

IM_DIM = 128 
VOL_DIM = 64

def inference(dataset_):
  is_train_pl = tf.placeholder(tf.bool)
  img_pl, _, _ = model.placeholder_inputs(BATCH_SIZE, IM_DIM, VOL_DIM)

  pred_reg_clr, pred_conf, pred_flow, pred_blended_clr = model.get_model(img_pl, is_train_pl)

  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  with tf.Session(config=config) as sess:
    model_path = os.path.join(TRAIN_DIR, "trained_models")
    ckpt = tf.train.get_checkpoint_state(model_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, ckpt.model_checkpoint_path)

    test_samples = dataset_.getTestSampleSize()

    for batch_idx in range(test_samples):
      imgs, view_names = dataset_.next_test_batch(batch_idx, 1) 

      feed_dict = {img_pl: imgs, is_train_pl: False}
      res_reg_clr, res_conf, res_flow, res_blended_clr = sess.run([pred_reg_clr, pred_conf, pred_flow, pred_blended_clr], feed_dict=feed_dict)

      instance_id = dataset_.getId('test', batch_idx)

      for i in range(len(view_names)):
        vol_reg_clr = res_reg_clr[i] # (vol_dim, vol_dim, vol_dim, 3)
        vol_conf = res_conf[i] # (vol_dim, vol_dim, vol_dim, 1)
        vol_flow = res_flow[i] # (vol_dim, vol_dim, vol_dim, 2)
        vol_blended_clr = res_blended_clr[i] # (vol_dim, vol_dim, vol_dim, 3)

        name_ = view_names[i][:-4] # xx.xxx.png

        save_path = os.path.join(OUTPUT_DIR, instance_id)
        if not os.path.exists(save_path): 
          os.makedirs(save_path)
        
        save_path_name = os.path.join(save_path, name_+".h5")
        if os.path.exists(save_path_name):
          os.remove(save_path_name)

        vol_ = np.concatenate((vol_reg_clr, vol_conf, vol_flow, vol_blended_clr), axis=-1) # (vol_dim, vol_dim, vol_dim, 9)

        h5_fout = h5py.File(save_path_name)
        h5_fout.create_dataset(
                'data', data=vol_,
                compression='gzip', compression_opts=4,
                dtype='float32')
        h5_fout.close()

        print batch_idx, instance_id, save_path_name


def main():
  test_dataset = dataset.Dataset(base_path=FLAGS.base_dir, 
                                  cat_id=FLAGS.cat_id, 
                                  data_list_path=FLAGS.data_list_path)
  inference(test_dataset)

if __name__ == '__main__':
  main()






