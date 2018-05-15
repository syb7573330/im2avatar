import tensorflow as tf
import sys
import numpy as np

sys.path.append('../utils')

import tf_utils

def placeholder_inputs(batch_size, im_dim, vol_dim):
  img_pl = tf.placeholder(tf.float32, shape=(batch_size, im_dim, im_dim, 3))
  vol_pl = tf.placeholder(tf.float32, shape=(batch_size, vol_dim, vol_dim, vol_dim, 1))
  return img_pl, vol_pl

def get_MSFE_cross_entropy_loss(pred, target):
  ''' Use loss = FPE + FNE, 
      FPE: negative samples, empty voxels in targets
      FNE: positive samples, non-empty voxels in targets

      ref: https://www-staff.it.uts.edu.au/~lbcao/publication/IJCNN15.wang.final.pdf 

      Args:
        pred: (batch, vol_dim, vol_dim, vol_dim, 1)
        target: (batch, vol_dim, vol_dim, vol_dim, 1), containing value = {0, 1}
      Rrturns:
        The total loss
  '''  
  cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred)
   
  # FPE
  fpe = tf.where(target < 0.5, cross_entropy_loss, tf.zeros_like(cross_entropy_loss))
  num_neg = tf.shape(tf.where(target < 0.5))[0] # int 
  num_neg = tf.to_float(num_neg)
  fpe = tf.reduce_sum(fpe) / num_neg


  # FNE
  fne = tf.where(target > 0.5, cross_entropy_loss, tf.zeros_like(cross_entropy_loss))
  num_pos = tf.shape(tf.where(target > 0.5))[0] # int
  num_pos = tf.to_float(num_pos)
  fne = tf.reduce_sum(fne) / num_pos

  loss = tf.square(fpe) + tf.square(fne)

  tf.add_to_collection('losses', loss)
  
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def get_model(imgs, is_training, weight_decay=0.0, bn_decay=None):
  """
      Args: 
        imgs: (batch_size, im_dim, im_dim, 3)
        is_training: a boolean placeholder.

      Return:
        shape: (batch_size, vol_dim, vol_dim, vol_dim, 1)
  """
  batch_size = imgs.get_shape()[0].value
  im_dim = imgs.get_shape()[1].value

  ########
  with tf.variable_scope('Encoding'):
    # (batch_size, 64, 64, 64)
    net = tf_utils.conv2d(imgs, 64, [7,7],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 32, 32, 64)
    net = tf_utils.conv2d(net, 64, [5,5],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 16, 16, 128)
    net = tf_utils.conv2d(net, 128, [5,5],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 8, 8, 128)
    net = tf_utils.conv2d(net, 128, [3,3],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 4, 4, 256)
    net = tf_utils.conv2d(net, 256, [3,3],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 1, 1, 512)
    net = tf_utils.conv2d(net, 512, [4,4],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)

  ########
  with tf.variable_scope('Latent_variable'):
    net = tf.reshape(net, [batch_size, 512])
    net = tf_utils.fully_connected(net, 512, scope="fc1", 
                    weight_decay=weight_decay, activation_fn=tf.nn.elu, 
                    bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_utils.fully_connected(net, 128*4*4*4, scope="fc2", 
                    weight_decay=weight_decay, activation_fn=tf.nn.elu, 
                    bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf.reshape(net, [batch_size, 4, 4, 4, 128])

  ########
  with tf.variable_scope('Decoding'):
    # (batch_size, 8, 8, 8, 64)
    net = tf_utils.conv3d_transpose(net, 64, [3, 3, 3], scope="deconv1",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
    # (batch_size, 16, 16, 16, 32)
    net = tf_utils.conv3d_transpose(net, 32, [3, 3, 3], scope="deconv2",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
    # (batch_size, 32, 32, 32, 32)
    net = tf_utils.conv3d_transpose(net, 32, [3, 3, 3], scope="deconv3",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
    # (batch_size, 64, 64, 64, 16)
    net = tf_utils.conv3d_transpose(net, 24, [3, 3, 3], scope="deconv4",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
    # (batch_size, 64, 64, 64, 1)
    net = tf_utils.conv3d(net, 1, [3, 3, 3], scope="deconv5",
                     stride=[1, 1, 1], padding='SAME',
                     weight_decay=weight_decay, activation_fn=None,
                     bn=True, bn_decay=bn_decay, is_training=is_training)

  return net


if __name__=='__main__':
  with tf.Graph().as_default():
    batch_size = 3
    im_dim = 128
    vol_dim = 64
    im_pl, vol_pl = placeholder_inputs(batch_size, im_dim, vol_dim)
    pred = get_model(im_pl, tf.constant(True))
    loss = get_MSFE_cross_entropy_loss(pred, vol_pl)
    
    print im_pl
    print vol_pl
    print pred
    print loss



